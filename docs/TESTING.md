# Guia de Testes do Cartola FC Optimizer

## 🛡️ Visão Geral

Este documento descreve a estratégia de testes, como executar os testes e como contribuir com novos testes.

## 📚 Índice

- [Instalação das Dependências](#instalação)
- [Executando os Testes](#executando-testes)
- [Estrutura dos Testes](#estrutura)
- [Escrevendo Novos Testes](#escrevendo-testes)
- [Boas Práticas](#boas-praticas)
- [Cobertura de Código](#cobertura)

---

## 📦 Instalação

```bash
# Instalar dependências de teste
pip install pytest pytest-cov pytest-mock

# Ou usando o Makefile
make install
```

---

## ▶️ Executando Testes

### Todos os Testes

```bash
# Usando pytest diretamente
pytest tests/ -v

# Usando Makefile
make test
```

### Testes com Cobertura

```bash
# Gerar relatório de cobertura
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Abrir relatório HTML
open htmlcov/index.html
```

### Testes Rápidos (sem slow tests)

```bash
pytest tests/ -v -m "not slow"

# Usando Makefile
make test-fast
```

### Testes Específicos

```bash
# Apenas um arquivo
pytest tests/test_validators.py -v

# Apenas uma classe
pytest tests/test_validators.py::TestAtletaModel -v

# Apenas um método
pytest tests/test_validators.py::TestAtletaModel::test_atleta_valido -v
```

### Testes por Categoria

```bash
# Apenas unitários
pytest tests/ -v -m unit

# Apenas integração
pytest tests/ -v -m integration
```

---

## 📁 Estrutura dos Testes

```
tests/
├── __init__.py              # Inicialização do pacote
├── conftest.py              # Fixtures compartilhadas
├── test_validators.py       # Testes de validação Pydantic
├── test_cache.py            # Testes do sistema de cache
├── test_exceptions.py       # Testes de exceções
├── test_api_client.py       # Testes do cliente de API
├── test_predictor.py        # Testes do preditor ML
├── test_optimizer.py        # Testes do otimizador genético
└── test_features.py         # Testes de feature engineering
```

---

## ✍️ Escrevendo Novos Testes

### Estrutura Básica

```python
import pytest
from src.module import ClassToTest


class TestClassName:
    """Descrição da suite de testes."""
    
    def test_scenario_description(self):
        """Testa cenário específico."""
        # Arrange (preparar)
        input_data = {...}
        expected_output = {...}
        
        # Act (executar)
        result = ClassToTest.method(input_data)
        
        # Assert (verificar)
        assert result == expected_output
    
    def test_error_handling(self):
        """Testa tratamento de erro."""
        with pytest.raises(ExpectedException):
            ClassToTest.method_that_fails()
```

### Usando Fixtures

```python
# conftest.py
@pytest.fixture
def sample_data():
    return {"key": "value"}


# test_file.py
def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### Parametrização

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    assert input * 2 == expected
```

### Marcadores (Markers)

```python
@pytest.mark.slow
def test_expensive_operation():
    """Teste que demora muito."""
    pass

@pytest.mark.integration
def test_api_integration():
    """Teste de integração com API."""
    pass

@pytest.mark.unit
def test_simple_function():
    """Teste unitário simples."""
    pass
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Testa com mock de dependência."""
    mock_api = Mock()
    mock_api.get_data.return_value = {"data": "test"}
    
    result = function_that_uses_api(mock_api)
    
    assert result["data"] == "test"
    mock_api.get_data.assert_called_once()


@patch('src.module.ExternalService')
def test_with_patch(mock_service):
    """Testa com patch de classe."""
    mock_service.return_value.method.return_value = "mocked"
    result = function_that_calls_service()
    assert result == "mocked"
```

---

## ✅ Boas Práticas

### 1. Nome dos Testes

- Use nomes descritivos: `test_should_return_error_when_input_is_invalid`
- Indique o que está sendo testado e o comportamento esperado
- Use inglês ou português consistentemente

### 2. Arrange-Act-Assert (AAA)

```python
def test_example():
    # Arrange - preparar dados e dependências
    input_data = create_test_data()
    
    # Act - executar a função/método
    result = function_under_test(input_data)
    
    # Assert - verificar resultado
    assert result.status == "success"
```

### 3. Um Conceito por Teste

- Cada teste deve verificar apenas um comportamento
- Evite testes gigantes que testam múltiplas coisas

### 4. Testes Independentes

- Testes não devem depender da ordem de execução
- Use fixtures para setup e teardown
- Limpe recursos após cada teste

### 5. Dados de Teste Realistas

- Use dados que refletem casos reais
- Inclua casos extremos (edge cases)
- Teste valores inválidos e limites

### 6. Mensagens de Erro Claras

```python
assert result > 0, f"Expected positive value, got {result}"
assert len(items) == 5, f"Expected 5 items, found {len(items)}"
```

---

## 📊 Cobertura de Código

### Metas de Cobertura

- **Mínimo aceitável**: 70%
- **Alvo recomendado**: 85%+
- **Ideal**: 90%+

### Verificar Cobertura

```bash
# Gerar relatório
pytest --cov=src --cov-report=term-missing

# Visualizar no navegador
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Interpretar Relatório

- **Linhas verdes**: Cobertas por testes
- **Linhas vermelhas**: Não cobertas
- **Linhas amarelas**: Parcialmente cobertas

### Melhorar Cobertura

1. Identificar módulos com baixa cobertura
2. Adicionar testes para caminhos não cobertos
3. Testar casos de erro e exceções
4. Cobrir branches condicionais (if/else)

---

## 🐞 Debug de Testes

### Executar com Verbose

```bash
pytest tests/ -vv
```

### Mostrar Print Statements

```bash
pytest tests/ -s
```

### Parar no Primeiro Erro

```bash
pytest tests/ -x
```

### Usar Debugger

```python
def test_with_debugger():
    import pdb; pdb.set_trace()
    result = function_to_debug()
    assert result == expected
```

---

## 🚀 CI/CD

Os testes são executados automaticamente no GitHub Actions:

- A cada push na branch main ou develop
- A cada pull request
- Verifica lint, types e testes
- Gera relatório de cobertura

Veja `.github/workflows/ci.yml` para detalhes.

---

## 📚 Recursos Adicionais

- [Documentação Pytest](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
