# Usar imagem oficial do Python 3.10 lite
FROM python:3.10-slim

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Sao_Paulo

# Instalar dependências de sistema mínimas para compilação (se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os requerimentos primeiro (aproveita cache do Docker)
# Se houver requirements.txt existente, usaremos ele. Senão instalamos o core.
COPY requirements.txt* ./

# Instalar dependências core da aplicação V2.2
RUN if [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pandas \
        numpy \
        scikit-learn \
        pulp \
        requests \
        python-dotenv \
        pyyaml; \
    fi

# Copiar todo o source code para dentro do container
COPY . .

# Expor a porta que a API do FastAPI utilizará
EXPOSE 8000

# Comando para rodar a aplicação via Uvicorn
CMD ["python", "-m", "uvicorn", "src.api.rest_app:app", "--host", "0.0.0.0", "--port", "8000"]
