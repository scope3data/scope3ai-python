class Scope3AIError(Exception):
    pass


class TracerInitializationError(Scope3AIError):
    """Tracer is initialized twice"""
    pass


class ModelingError(Scope3AIError):
    """Operation or computation not allowed"""
    pass
