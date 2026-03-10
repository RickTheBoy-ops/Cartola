"""Exceções customizadas para tratamento de erros específicos."""


class CartolaBaseException(Exception):
    """Exceção base para todas as exceções do sistema."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Detalhes: {self.details}"
        return self.message


# ========== Exceções de API ==========

class APIException(CartolaBaseException):
    """Exceção base para erros de API."""
    pass


class APIConnectionError(APIException):
    """Erro de conexão com a API do Cartola."""
    pass


class APITimeoutError(APIException):
    """Timeout ao tentar conectar com a API."""
    pass


class APIRateLimitError(APIException):
    """Rate limit da API excedido."""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, {'retry_after': retry_after})
        self.retry_after = retry_after


class APIAuthenticationError(APIException):
    """Erro de autenticação na API."""
    pass


class APIResponseError(APIException):
    """Resposta inválida da API."""
    pass


# ========== Exceções de Dados ==========

class DataException(CartolaBaseException):
    """Exceção base para erros de dados."""
    pass


class DataValidationError(DataException):
    """Dados não passaram na validação."""
    pass


class DataNotFoundError(DataException):
    """Dados esperados não foram encontrados."""
    pass


class InsufficientDataError(DataException):
    """Dados insuficientes para operação."""
    
    def __init__(self, message: str, required: int, available: int):
        super().__init__(message, {'required': required, 'available': available})
        self.required = required
        self.available = available


# ========== Exceções de Mercado ==========

class MercadoException(CartolaBaseException):
    """Exceção base para erros relacionados ao mercado."""
    pass


class MercadoFechadoError(MercadoException):
    """Mercado está fechado."""
    pass


class MercadoEmManutencaoError(MercadoException):
    """Mercado em manutenção."""
    pass


# ========== Exceções de Otimização ==========

class OptimizationException(CartolaBaseException):
    """Exceção base para erros de otimização."""
    pass


class InvalidFormationError(OptimizationException):
    """Formação tática inválida."""
    pass


class InsufficientBudgetError(OptimizationException):
    """Patrimônio insuficiente para formar time."""
    
    def __init__(self, message: str, required: float, available: float):
        super().__init__(message, {'required': required, 'available': available})
        self.required = required
        self.available = available


class NoValidTeamError(OptimizationException):
    """Nenhum time válido encontrado com as restrições."""
    pass


# ========== Exceções de ML ==========

class MLException(CartolaBaseException):
    """Exceção base para erros de Machine Learning."""
    pass


class ModelNotTrainedError(MLException):
    """Modelo não foi treinado ainda."""
    pass


class ModelLoadError(MLException):
    """Erro ao carregar modelo salvo."""
    pass


class FeatureEngineeringError(MLException):
    """Erro durante feature engineering."""
    pass


class PredictionError(MLException):
    """Erro durante predição."""
    pass


# ========== Funções Auxiliares ==========

def handle_api_error(response_code: int, response_text: str = ""):
    """Converte código HTTP em exceção apropriada.
    
    Args:
        response_code: Código HTTP da resposta
        response_text: Texto da resposta (opcional)
        
    Raises:
        APIException apropriada
    """
    error_map = {
        400: (APIResponseError, "Requisição inválida"),
        401: (APIAuthenticationError, "Falha na autenticação"),
        403: (APIAuthenticationError, "Acesso negado"),
        404: (DataNotFoundError, "Recurso não encontrado"),
        429: (APIRateLimitError, "Rate limit excedido"),
        500: (APIException, "Erro interno do servidor"),
        502: (APIConnectionError, "Bad Gateway"),
        503: (APIException, "Serviço indisponível"),
        504: (APITimeoutError, "Gateway timeout"),
    }
    
    exception_class, default_message = error_map.get(
        response_code,
        (APIException, f"Erro HTTP {response_code}")
    )
    
    message = response_text if response_text else default_message
    raise exception_class(message, {'status_code': response_code})
