class CartolaBaseError(Exception):
    """Exceção base para todo o ecossistema Cartola"""
    pass

class APIConnectionError(CartolaBaseError):
    """Lançada quando há problemas na comunicação com a API da Globo"""
    pass

class DataValidationError(CartolaBaseError):
    """Lançada quando os dados recebidos são inválidos ou corrompidos"""
    pass

class OptimizationError(CartolaBaseError):
    """Lançada quando o otimizador não consegue gerar time viável"""
    pass

class ConfigError(CartolaBaseError):
    """Lançada quando configurações yaml / env vars são inválidas"""
    pass
