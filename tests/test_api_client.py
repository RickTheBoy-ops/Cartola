import pytest
from unittest.mock import patch, Mock
from src.api.client import CartolaAPIClient, CartolaAPIError
from src.utils.exceptions import APIConnectionError

@pytest.fixture
def mock_client():
    # Cria uma instância sem disparar autenticação real
    with patch('src.api.client.CartolaAPIClient.authenticate'):
        client = CartolaAPIClient(email="fake@email.com", password="fake")
        client.glb_token = "fake_token"
        # Limpar cache para evitar interferência entre testes
        from src.utils.cache import api_cache
        api_cache._cache.clear()
        return client

@patch('src.api.client.requests.Session.request')
def test_get_mercado_status_success(mock_request, mock_client):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "aberto", "rodada_atual": 10}
    mock_response.content = b'{"status": "aberto", "rodada_atual": 10}'
    mock_request.return_value = mock_response

    result = mock_client.get_mercado_status()
    assert result['status'] == 'aberto'
    assert result['rodada_atual'] == 10

@patch('src.api.client.requests.Session.request')
def test_get_mercado_status_unauthorized(mock_request, mock_client):
    mock_response = Mock()
    mock_response.status_code = 401
    
    # Criar um erro HTTP embutido
    import requests
    http_error = requests.exceptions.HTTPError()
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error
    
    mock_request.return_value = mock_response

    with pytest.raises(CartolaAPIError) as excinfo:
        mock_client.get_mercado_status()

    assert "Não autenticado" in str(excinfo.value)
