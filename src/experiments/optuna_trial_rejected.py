"""A completed trial whose diagnostics make it ineligible for optimization."""


class OptunaTrialRejected(RuntimeError):
    """Mark a trial FAIL while retaining its complete diagnostic artifacts."""
