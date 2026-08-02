from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import structlog

from app.adapters.cache import keys as cache_keys
from app.domain.auth import AuthContext
from app.domain.enums import Role
from app.domain.policies.rbac import require_tool
from app.application.planning.tool_schemas import (
    RESUME_SEARCH_DESCRIPTION,
    RESUME_SEARCH_INPUT_SCHEMA,
)
from app.domain.tools.base import SourceRef, ToolMeta, ToolResult
from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingClient
from app.ports.vector_store import VectorStore
from app.tools.resume_search.attributes import ResumeAttribute, resolve_purpose
from app.tools.resume_search.birthday import today_utc
from app.tools.resume_search.context_builder import build_structured_context
from app.tools.resume_search.entity_resolution import Resolution, resolve_employees
from app.tools.resume_search.fusion import DEFAULT_K
from app.tools.resume_search.reranker import rerank
from app.tools.resume_search.skills import extract_skill, filter_hits_for_skill

log = structlog.get_logger(__name__)

# Cosine / lexical floor for generic skill RAG before publishing employee_ids.
_GENERIC_MIN_SCORE = 0.32
# Bump when attribute extractors/answers change so Redis cannot serve stale prose.
_ATTRIBUTE_CACHE_VERSION = "attr-v3"


def _score_gate_hits(
    hits: list[dict[str, Any]], *, min_score: float
) -> list[dict[str, Any]]:
    """Drop weak semantic hits so they cannot become a confident cohort."""
    kept: list[dict[str, Any]] = []
    for h in hits:
        try:
            score = float(h.get("score") if h.get("score") is not None else 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= min_score:
            kept.append(h)
    return kept


def _as_employee_ids(raw: Any) -> list[str]:
    """Normalise bound ids from a list, a single UUID, or an employee payload."""
    if raw is None or raw is False:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, dict):
        if raw.get("id") is not None:
            return [str(raw["id"])]
        emp = raw.get("employee")
        if isinstance(emp, dict) and emp.get("id") is not None:
            return [str(emp["id"])]
        if isinstance(raw.get("employee_ids"), list):
            return _as_employee_ids(raw["employee_ids"])
        return []
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for x in raw:
            if isinstance(x, dict) and x.get("id") is not None:
                out.append(str(x["id"]))
            elif x is not None and str(x).strip():
                out.append(str(x))
        return out
    return [str(raw)]


def _hit_employee_name(hit: dict[str, Any]) -> str:
    return str(
        hit.get("employee_name")
        or (hit.get("metadata") or {}).get("employee_name")
        or ""
    ).strip()


def _name_hit_matches(query: str, hit: dict[str, Any]) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    name = _hit_employee_name(hit).lower()
    if not name:
        return False
    if q == name or q in name or name in q:
        return True
    tokens = [t for t in q.split() if len(t) > 1]
    return bool(tokens) and all(t in name for t in tokens)


def _seconds_to_utc_midnight(now: datetime | None = None) -> int:
    """TTL that expires at the next UTC midnight.

    "Whose birthday is today" is only true for one day, so a cached answer must
    never survive the date change that invalidates it.
    """
    moment = now or datetime.now(UTC)
    tomorrow = datetime.combine(
        moment.date() + timedelta(days=1), time.min, tzinfo=UTC
    )
    return max(1, int((tomorrow - moment).total_seconds()))


class ResumeSearchTool:
    def __init__(
        self,
        *,
        embeddings: EmbeddingClient,
        vector_store: VectorStore,
        cache: CachePort,
        top_k: int = 30,
        rerank_top_k: int = 5,
        cache_ttl: int = 1800,
        rrf_k: int = DEFAULT_K,
        name_similarity: float = 0.35,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._cache = cache
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._cache_ttl = cache_ttl
        self._rrf_k = rrf_k
        self._name_similarity = name_similarity
        self._meta = ToolMeta(
            name="resume_search",
            description=RESUME_SEARCH_DESCRIPTION,
            input_schema=RESUME_SEARCH_INPUT_SCHEMA,
            permissions=list(Role),
            estimated_latency_ms=400,
            cache_policy="retrieval",
        )

    @property
    def meta(self) -> ToolMeta:
        return self._meta

    async def run(self, params: dict[str, Any], *, auth: AuthContext) -> ToolResult:
        require_tool(auth, "resume_search")
        question = params.get("question") or params.get("query") or ""
        purpose = params.get("purpose") or ""
        # Optional cohort filter (SQL→RAG): only keep hits in this employee set.
        # Bindings may hand us a list, a single UUID, or an employee-tool payload.
        scope_ids = set(_as_employee_ids(params.get("employee_ids")))
        if scope_ids:
            params["employee_ids"] = sorted(scope_ids)
        scope_key = ",".join(sorted(scope_ids)[:80]) if scope_ids else ""
        attribute_request = resolve_purpose(purpose)
        skill_key = str(params.get("skill") or "").strip().lower()
        key = cache_keys.build(
            "retrieval",
            auth,
            q=question,
            model=self._embeddings.model_name,
            top_k=self._top_k,
            scope=scope_key,
            purpose=purpose,
            skill=skill_key,
            # Attribute answers are date-sensitive ("today", "turning 41") and are
            # shaped by these params, so two questions about the same person or
            # cohort must not share a cache entry.
            day=today_utc().isoformat() if attribute_request else "",
            shape=self._answer_shape(params) if attribute_request else "",
            attr_v=_ATTRIBUTE_CACHE_VERSION if attribute_request else "",
        )
        cached = await self._cache.get(key)
        if cached is not None:
            return ToolResult(data=cached, confidence=0.85, cache_hit=True)

        try:
            if attribute_request is not None:
                attribute, mode = attribute_request
                runners = {
                    "cohort": self._attribute_cohort,
                    "facet": self._attribute_facet,
                    "person": self._attribute_person,
                }
                result = await runners[mode](attribute, params)
                await self._cache.set(
                    key, result.data, ttl_seconds=_seconds_to_utc_midnight()
                )
                return result

            hits: list[dict[str, Any]] = []

            # Name-scoped write flows resolve one person from resume metadata.
            name_resolve = purpose == "status_resolve"
            if name_resolve:
                named = await self._vector_store.search_by_employee_name(
                    question, limit=max(40, self._top_k)
                )
                hits.extend(named)

            emb = await self._embeddings.embed(question)
            fetch_k = self._top_k * 5 if name_resolve else (
                self._top_k * 3 if scope_ids else self._top_k
            )
            semantic = await self._vector_store.similarity_search(
                embedding=emb, top_k=fetch_k, tenant_id=auth.tenant_id
            )
            hits.extend(semantic)

            # Dedupe by chunk id
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for h in hits:
                cid = str(h.get("id") or "")
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                deduped.append(h)
            hits = deduped

            if scope_ids:
                hits = [h for h in hits if str(h.get("employee_id")) in scope_ids]

            purpose = str(params.get("purpose") or "").strip().lower()
            if name_resolve:
                named_hits = [h for h in hits if _name_hit_matches(question, h)]
                if named_hits:
                    hits = named_hits
                rerank_k = max(self._rerank_top_k, 20)
            elif purpose == "skill":
                # Skill counts/lists need broad recall — default rerank_top_k=5
                # was collapsing "how many know Python?" to a handful of hits.
                rerank_k = max(self._rerank_top_k, self._top_k)
            else:
                rerank_k = self._rerank_top_k

            ranked = rerank(question, hits, rerank_k)
            skill = extract_skill(
                question, explicit=str(params.get("skill") or "").strip() or None
            )
            if name_resolve:
                ranked = [h for h in ranked if _name_hit_matches(question, h)] or ranked
            elif skill and purpose in {"", "skill"}:
                # RAG recall stays semantic; require the chunk text to actually
                # mention the skill so Docker neighbors cannot join Kubernetes.
                # Also covers heuristic skill plans that omit purpose=skill.
                before = len(ranked)
                ranked = filter_hits_for_skill(ranked, skill)
                if before and not ranked:
                    log.info(
                        "skill_lexical_filter_empty",
                        skill=skill,
                        question=question,
                        pre_filter_hits=before,
                    )
            else:
                # Score-gate generic semantic hits before materializing a "them" cohort.
                ranked = _score_gate_hits(ranked, min_score=_GENERIC_MIN_SCORE)

            context = build_structured_context(ranked)
            if not name_resolve and not ranked:
                context = {
                    **context,
                    "employee_ids": [],
                    "hits": [],
                    "score_gated": True,
                }
            await self._cache.set(key, context, ttl_seconds=self._cache_ttl)
            return ToolResult(
                data=context,
                confidence=0.85 if ranked else 0.2,
                sources=self._sources(ranked),
            )
        except Exception as exc:
            return ToolResult(
                data={"hits": [], "employee_ids": []},
                confidence=0.0,
                error=str(exc),
                degraded=True,
            )

    async def _attribute_person(
        self,
        attribute: ResumeAttribute,
        params: dict[str, Any],
    ) -> ToolResult:
        """One person's resume-only fact: resolve who, retrieve the section, extract."""
        name = str(params.get("name") or params.get("question") or "")
        # When a prior node already fixed the person (pronoun binding), skip name
        # resolution so a namesake cannot steal the answer.
        bound_ids = _as_employee_ids(params.get("employee_ids"))
        if bound_ids:
            from app.tools.resume_search.entity_resolution import (
                Resolution,
                ResolvedEmployee,
            )

            resolution = Resolution(
                candidates=[
                    ResolvedEmployee(employee_id=eid, name=name, score=1.0)
                    for eid in bound_ids
                ],
                via="bound",
            )
        else:
            resolution = await resolve_employees(
                query=name,
                store=self._vector_store,
                embeddings=self._embeddings,
                exclude_sections=attribute.sections,
                min_similarity=self._name_similarity,
                rrf_k=self._rrf_k,
                # The planner supplies a person name, so an unmatched name must return
                # nothing rather than the best-ranked stranger. Callers that pass a
                # descriptive reference instead opt into resolving from prose.
                descriptive_fallback=str(params.get("reference") or "name")
                == "descriptive",
            )
        hits: list[dict[str, Any]] = []
        if resolution.candidates:
            hits = await self._vector_store.fetch_section_chunks(
                attribute.sections,
                employee_ids=resolution.employee_ids,
                content_hints=attribute.content_hints,
                limit=max(40, len(resolution.candidates) * 4),
            )
        # Weak fuzzy / fused identity → clarify instead of answering a namesake.
        if resolution.via == "fuzzy":
            best = resolution.candidates[0] if resolution.candidates else None
            if best is None or float(best.name_score or 0.0) < 0.55:
                return ToolResult(
                    data={
                        "clarify": (
                            resolution.note
                            or f'I\'m not sure who "{name}" refers to. '
                            "Please use a fuller name."
                        ),
                        "employee_ids": [],
                        "hits": [],
                    },
                    confidence=0.35,
                )
            if len(resolution.candidates) > 1:
                options = [
                    c.name or c.employee_id for c in resolution.candidates[:8] if c.name
                ]
                return ToolResult(
                    data={
                        "clarify": (
                            f'I found multiple close matches for "{name}". '
                            "Which one did you mean:\n"
                            + "\n".join(f"- {opt}" for opt in options)
                        ),
                        "employee_ids": [],
                        "hits": [],
                    },
                    confidence=0.4,
                )
        if resolution.via == "fused" and str(params.get("reference") or "name") != "descriptive":
            return ToolResult(
                data={
                    "clarify": (
                        f'I couldn\'t confidently match "{name}" to one employee. '
                        "Please use a fuller name."
                    ),
                    "employee_ids": [],
                    "hits": [],
                },
                confidence=0.35,
            )

        facts = attribute.facts_from_hits(hits)
        log.info(
            "resume_attribute_person",
            attribute=attribute.name,
            query=name,
            resolved_via=resolution.via,
            candidates=len(resolution.candidates),
            chunks=len(hits),
            facts=len(facts),
        )

        today = today_utc()
        answer = attribute.answer_person(
            facts,
            name_asked=name,
            wants_age=bool(params.get("wants_age")),
            wants_wish=bool(params.get("wants_wish")),
            skill=params.get("skill"),
            today=today,
            note=resolution.note,
        )
        return self._attribute_result(facts, hits, answer, today, resolution=resolution)

    async def _attribute_cohort(
        self,
        attribute: ResumeAttribute,
        params: dict[str, Any],
    ) -> ToolResult:
        """Cohort question: retrieve the whole section slice, then filter in Python.

        Completeness, not ranking, is the requirement here — a top-k embedding
        search would silently sample the corpus.
        """
        # Distinguish "not scoped" from "scoped to nobody". An explicit empty
        # employee_ids list must never fall through to the whole corpus.
        scoped = "employee_ids" in params
        scope_ids = _as_employee_ids(params.get("employee_ids"))
        if scoped and not scope_ids:
            return ToolResult(
                data={
                    "hits": [],
                    "employee_ids": [],
                    "facts": [],
                    "answer": None,
                    "score_gated": True,
                },
                confidence=0.2,
            )
        chunks = await self._vector_store.fetch_section_chunks(
            attribute.sections,
            employee_ids=scope_ids if scoped else None,
            content_hints=attribute.content_hints,
        )
        facts = attribute.facts_from_hits(chunks)
        today = today_utc()
        # Coverage compares against the whole corpus, which is only meaningful
        # when the whole corpus was in scope.
        coverage = (
            None
            if scoped
            else (len(facts), await self._vector_store.count_indexed_employees())
        )

        matched = (
            attribute.select_cohort(facts, params=params, today=today)
            if attribute.select_cohort
            else facts
        )
        matched_ids = {f.employee_id for f in matched}
        relevant = [h for h in chunks if str(h.get("employee_id")) in matched_ids]

        answer = None
        if attribute.answer_cohort is not None:
            month = params.get("month")
            cohort_kwargs: dict[str, Any] = {
                "scope": str(params.get("scope") or "today"),
                "month": int(month) if month else None,
                "today": today,
                "coverage": coverage,
                "language": params.get("language"),
                "certification": params.get("certification"),
                "among_prior": bool(scoped and scope_ids),
            }
            # Prefer the filtered cohort when the attribute publishes ids.
            # Attribute answer_cohort callables ignore unknown kwargs via **_ignored.
            answer_facts = matched if attribute.publishes_cohort_ids else facts
            answer = attribute.answer_cohort(answer_facts, **cohort_kwargs)
        log.info(
            "resume_attribute_cohort",
            attribute=attribute.name,
            scope=params.get("scope"),
            place=params.get("city") or params.get("country"),
            scoped_ids=len(scope_ids),
            chunks=len(chunks),
            facts=len(facts),
            matched=len(matched),
            indexed_employees=coverage[1] if coverage else None,
        )

        # When the cohort itself is the result, only matching employees may be
        # published: an unrelated id here becomes a wrong row in the next node.
        cited = relevant if attribute.publishes_cohort_ids else (relevant or chunks[:20])
        return self._attribute_result(
            matched if attribute.publishes_cohort_ids else facts,
            cited,
            answer,
            today,
            coverage=coverage,
        )

    async def _attribute_facet(
        self,
        attribute: ResumeAttribute,
        params: dict[str, Any],
    ) -> ToolResult:
        """Aggregate a resume-only field ("how many different cities").

        Deliberately publishes no employee ids: an aggregate is not a cohort, and
        letting 100 people become the active "them" would wreck the next turn.
        """
        dimension = str(params.get("facet") or "")
        if dimension not in attribute.facets:
            return ToolResult(
                data={"hits": [], "employee_ids": [], "rows": []},
                confidence=0.0,
                error=f"{attribute.name} cannot be aggregated by {dimension!r}",
                degraded=True,
            )
        chunks = await self._vector_store.fetch_section_chunks(
            attribute.sections,
            content_hints=attribute.content_hints,
        )
        facts = attribute.facts_from_hits(chunks)
        today = today_utc()
        values = sorted(
            {
                str(value)
                for value in (f.to_dict(today).get(dimension) for f in facts)
                if value
            }
        )
        total = await self._vector_store.count_indexed_employees()
        log.info(
            "resume_attribute_facet",
            attribute=attribute.name,
            facet=dimension,
            values=len(values),
            facts=len(facts),
            indexed_employees=total,
        )
        return ToolResult(
            data={
                "hits": [],
                "employee_ids": [],
                # Shaped like a SQL DISTINCT result so the formatter and the
                # facet memory in context_updates need no special case.
                "rows": [{dimension: value} for value in values],
                "count": len(values),
                "facet": dimension,
                "coverage": {"covered": len(facts), "total": total},
            },
            confidence=0.9,
        )

    def _attribute_result(
        self,
        facts: list[Any],
        hits: list[dict[str, Any]],
        answer: str | None,
        today: date,
        *,
        resolution: Resolution | None = None,
        coverage: tuple[int, int] | None = None,
    ) -> ToolResult:
        context = build_structured_context(hits)
        context["facts"] = [f.to_dict(today) for f in facts]
        # A resolution node carries no prose: the tool that owns the remaining
        # data writes the sentence, and an "answer" here would pre-empt it.
        if answer is not None:
            context["answer"] = answer
        if resolution is not None:
            context["resolved_via"] = resolution.via
        if coverage is not None:
            context["coverage"] = {"covered": coverage[0], "total": coverage[1]}
        return ToolResult(data=context, confidence=0.9, sources=self._sources(hits))

    @staticmethod
    def _answer_shape(params: dict[str, Any]) -> str:
        """The params that change the wording, not just the retrieval."""
        return "|".join(
            str(params.get(field) or "")
            for field in (
                "name",
                "reference",
                "scope",
                "month",
                "wants_age",
                "wants_wish",
                # Without these, "employees in Berlin" and "employees in Dubai"
                # would share a cache entry.
                "city",
                "country",
                "facet",
            )
        )

    @staticmethod
    def _sources(hits: list[dict[str, Any]]) -> list[SourceRef]:
        return [
            SourceRef(
                kind="resume_chunk",
                ref=str(h.get("id", "")),
                label=str(h.get("employee_id")),
            )
            for h in hits
        ]
