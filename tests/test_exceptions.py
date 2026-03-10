"""Testes para exceções customizadas."""

import pytest
from src.utils.exceptions import (
    CartolaBaseException,
    APIConnectionError,
    APITimeoutError,
    APIRateLimitError,
    APIAuthenticationError,
    DataValidationError,
    InsufficientDataError,
    MercadoFechadoError,
    InvalidFormationError,
    InsufficientBudgetError,
    ModelNotTrainedError,
    handle_api_error
)


class TestBaseException:
    """Testes para exceção base."""
    
    def test_exception_com_mensagem(self):
        """Testa exceção com mensagem simples."""
        exc = CartolaBaseException("Erro de teste")
        assert str(exc) == "Erro de teste"
    
    def test_exception_com_detalhes(self):
        """Testa exceção com detalhes."""
        exc = CartolaBaseException(
            "Erro com detalhes",
            details={'code': 500, 'info': 'Teste'}
        )
        assert "Erro com detalhes" in str(exc)
        assert "Detalhes" in str(exc)
        assert exc.details['code'] == 500


class TestAPIExceptions:
    """Testes para exceções de API."""
    
    def test_connection_error(self):
        """Testa erro de conexão."""
        with pytest.raises(APIConnectionError) as exc_info:
            raise APIConnectionError("Falha ao conectar")
        
        assert "Falha ao conectar" in str(exc_info.value)
    
    def test_timeout_error(self):
        """Testa erro de timeout."""
        with pytest.raises(APITimeoutError) as exc_info:
            raise APITimeoutError("Timeout após 30s")
        
        assert "Timeout" in str(exc_info.value)
    
    def test_rate_limit_error(self):
        """Testa erro de rate limit."""
        exc = APIRateLimitError("Rate limit excedido", retry_after=60)
        assert exc.retry_after == 60
        assert exc.details['retry_after'] == 60
    
    def test_authentication_error(self):
        """Testa erro de autenticação."""
        with pytest.raises(APIAuthenticationError):
            raise APIAuthenticationError("Credenciais inválidas")


class TestDataExceptions:
    """Testes para exceções de dados."""
    
    def test_validation_error(self):
        """Testa erro de validação."""
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError("Dados inválidos")
        
        assert "inválidos" in str(exc_info.value)
    
    def test_insufficient_data_error(self):
        """Testa erro de dados insuficientes."""
        exc = InsufficientDataError(
            "Dados insuficientes",
            required=100,
            available=50
        )
        
        assert exc.required == 100
        assert exc.available == 50
        assert exc.details['required'] == 100


class TestMercadoExceptions:
    """Testes para exceções de mercado."""
    
    def test_mercado_fechado(self):
        """Testa erro de mercado fechado."""
        with pytest.raises(MercadoFechadoError) as exc_info:
            raise MercadoFechadoError("Mercado fechado para escalações")
        
        assert "fechado" in str(exc_info.value)


class TestOptimizationExceptions:
    """Testes para exceções de otimização."""
    
    def test_invalid_formation(self):
        """Testa erro de formação inválida."""
        with pytest.raises(InvalidFormationError):
            raise InvalidFormationError("Formação 5-5-5 inválida")
    
    def test_insufficient_budget(self):
        """Testa erro de orçamento insuficiente."""
        exc = InsufficientBudgetError(
            "Orçamento insuficiente",
            required=150.0,
            available=100.0
        )
        
        assert exc.required == 150.0
        assert exc.available == 100.0


class TestMLExceptions:
    """Testes para exceções de ML."""
    
    def test_model_not_trained(self):
        """Testa erro de modelo não treinado."""
        with pytest.raises(ModelNotTrainedError) as exc_info:
            raise ModelNotTrainedError("Modelo não foi treinado")
        
        assert "treinado" in str(exc_info.value)


class TestHandleAPIError:
    """Testes para handler de erros de API."""
    
    def test_status_400(self):
        """Testa tratamento de status 400."""
        with pytest.raises(Exception):  # APIResponseError
            handle_api_error(400, "Bad Request")
    
    def test_status_401(self):
        """Testa tratamento de status 401."""
        with pytest.raises(APIAuthenticationError):
            handle_api_error(401, "Unauthorized")
    
    def test_status_404(self):
        """Testa tratamento de status 404."""
        with pytest.raises(Exception):  # DataNotFoundError
            handle_api_error(404, "Not Found")
    
    def test_status_429(self):
        """Testa tratamento de status 429."""
        with pytest.raises(APIRateLimitError):
            handle_api_error(429, "Rate Limit Exceeded")
    
    def test_status_500(self):
        """Testa tratamento de status 500."""
        with pytest.raises(Exception):  # APIException
            handle_api_error(500, "Internal Server Error")
    
    def test_status_504(self):
        """Testa tratamento de status 504."""
        with pytest.raises(APITimeoutError):
            handle_api_error(504, "Gateway Timeout")
