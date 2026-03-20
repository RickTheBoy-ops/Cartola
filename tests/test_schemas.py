import pytest
from src.api.schemas import AtletaModel, OptimizationRequest

def test_atleta_model_valid():
    try:
        atleta = AtletaModel(
            atleta_id=100,
            apelido="Pelé",
            preco=15.0,
            clube_id=20,
            posicao_id=5
        )
        assert atleta.apelido == "Pelé"
        assert atleta.preco == 15.0
    except ValueError:
        pytest.fail("Validação correta foi rejeitada.")

def test_atleta_model_invalid_price():
    with pytest.raises(ValueError) as exc:
        AtletaModel(
            atleta_id=101,
            apelido="Negativo",
            preco=-5.0, # Inválido
            clube_id=20,
            posicao_id=5
        )
    assert "Preço deve ser positivo" in str(exc.value)

def test_atleta_model_invalid_position():
    with pytest.raises(ValueError) as exc:
        AtletaModel(
            atleta_id=102,
            apelido="Lateral 7",
            preco=5.0,
            clube_id=20,
            posicao_id=7 # Máx é 6
        )
    assert "Posição deve ser entre 1 e 6" in str(exc.value)

def test_optimization_request_defaults():
    req = OptimizationRequest()
    assert req.strategy == "genetic"  # schema atual usa 'genetic'
    assert req.budget == 110.0         # padrão no schema é 110 cartoletas
    assert req.formation == "4-3-3"

def test_optimization_request_invalid_budget():
    with pytest.raises(ValueError) as exc:
        OptimizationRequest(budget=-10)
    assert "maior que zero" in str(exc.value)
