from __future__ import annotations

def setup_otel(service_name: str = "hrmind") -> None:
    # Optional OTel bootstrap; safe no-op if SDK misconfigured
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)
    except Exception:
        pass
