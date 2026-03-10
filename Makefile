.PHONY: help install test lint format clean run docker-build docker-up docker-down

# Variáveis
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy

# Cores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Mostra esta ajuda
	@echo "$(GREEN)Comandos disponíveis:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instala todas as dependências
	@echo "$(GREEN)Instalando dependências...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-mock black flake8 mypy isort bandit
	@echo "$(GREEN)✓ Instalação concluída!$(NC)"

install-dev: install ## Instala dependências de desenvolvimento
	$(PIP) install -e .
	@echo "$(GREEN)✓ Ambiente de desenvolvimento configurado!$(NC)"

test: ## Executa todos os testes
	@echo "$(GREEN)Executando testes...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Testes concluídos! Relatório em htmlcov/index.html$(NC)"

test-fast: ## Executa testes rápidos (sem coverage)
	@echo "$(GREEN)Executando testes rápidos...$(NC)"
	$(PYTEST) tests/ -v -m "not slow"

test-unit: ## Executa apenas testes unitários
	$(PYTEST) tests/ -v -m unit

test-integration: ## Executa apenas testes de integração
	$(PYTEST) tests/ -v -m integration

lint: ## Verifica qualidade do código
	@echo "$(GREEN)Verificando código com flake8...$(NC)"
	$(FLAKE8) src tests --max-line-length=100 --exclude=__pycache__
	@echo "$(GREEN)Verificando tipos com mypy...$(NC)"
	$(MYPY) src --ignore-missing-imports
	@echo "$(GREEN)✓ Lint concluído!$(NC)"

format: ## Formata o código automaticamente
	@echo "$(GREEN)Formatando código com Black...$(NC)"
	$(BLACK) src tests
	@echo "$(GREEN)Organizando imports com isort...$(NC)"
	isort src tests
	@echo "$(GREEN)✓ Formatação concluída!$(NC)"

format-check: ## Verifica formatação sem modificar
	$(BLACK) --check src tests
	isort --check-only src tests

security: ## Verifica vulnerabilidades de segurança
	@echo "$(GREEN)Verificando segurança com Bandit...$(NC)"
	bandit -r src -f screen

clean: ## Remove arquivos temporários e cache
	@echo "$(YELLOW)Limpando arquivos temporários...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf build/ dist/
	rm -rf data/cache/*.cache
	@echo "$(GREEN)✓ Limpeza concluída!$(NC)"

run: ## Executa o sistema principal
	@echo "$(GREEN)Executando Cartola FC Optimizer...$(NC)"
	$(PYTHON) main.py

run-streamlit: ## Executa dashboard Streamlit
	@echo "$(GREEN)Iniciando dashboard Streamlit...$(NC)"
	streamlit run app.py

docker-build: ## Builda imagens Docker
	@echo "$(GREEN)Buildando imagens Docker...$(NC)"
	docker-compose build

docker-up: ## Inicia containers Docker
	@echo "$(GREEN)Iniciando containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Containers iniciados!$(NC)"

docker-down: ## Para containers Docker
	@echo "$(YELLOW)Parando containers...$(NC)"
	docker-compose down

docker-logs: ## Mostra logs dos containers
	docker-compose logs -f

backup-db: ## Faz backup do banco de dados
	@echo "$(GREEN)Criando backup do banco...$(NC)"
	mkdir -p backups
	cp data/raw/cartola.db backups/cartola_$(shell date +%Y%m%d_%H%M%S).db
	@echo "$(GREEN)✓ Backup criado!$(NC)"

check: lint test ## Executa lint e testes

ci: install lint test ## Simula pipeline CI localmente
	@echo "$(GREEN)✓ Pipeline CI completo!$(NC)"

all: clean install format lint test ## Executa todos os comandos
	@echo "$(GREEN)✓ Todos os comandos executados com sucesso!$(NC)"
