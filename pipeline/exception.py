class PipelineError(Exception):
    """Base class for all pipeline errors."""
    pass

class InsufficientDataError(PipelineError):
    """Raised when the dataset has insufficient data for processing."""
    pass

class FitBeforeTransformError(PipelineError):
    """Raised when transform() is called before fit()."""
    pass

class UnsupportedModelTypeError(PipelineError):
    """Raised when an unsupported model type is specified in the config."""
    pass

class SerializationError(PipelineError):
    """Raised when saving or loading a fitted pipeline fails."""
    pass