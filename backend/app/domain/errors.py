class DomainError(Exception):
    def __init__(self, message: str, *, code: str = "domain_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class NotFoundError(DomainError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, code="not_found")


class ForbiddenError(DomainError):
    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(message, code="forbidden")


class ValidationFailedError(DomainError):
    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message, code="validation_failed")


class PlanInvalidError(DomainError):
    def __init__(self, message: str = "Execution plan is invalid") -> None:
        super().__init__(message, code="plan_invalid")


class ToolFailedError(DomainError):
    def __init__(self, message: str = "Tool execution failed") -> None:
        super().__init__(message, code="tool_failed")


class CacheKeyError(DomainError):
    def __init__(self, message: str = "Invalid cache key material") -> None:
        super().__init__(message, code="cache_key_error")
