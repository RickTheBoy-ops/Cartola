# 🏆 Cartola FC - Ultimate Optimizer (AI-Powered)

Sistema profissional de análise, predição e otimização para Cartola FC utilizando Machine Learning e Algoritmos Genéticos.

---

## 📁 Estrutura do Projeto

```
cartola/
├── src/
│   ├── api/
│   │   └── client.py              # Cliente HTTP robusto com Retry e Rate Limiting
│   ├── data/
│   │   └── collector.py           # ETL e persistência em SQLite
│   ├── ml/
│   │   ├── features.py            # Feature Engineering avançado
│   │   ├── predictor.py           # Modelos de ML (Random Forest/GB)
│   │   └── optimizer.py           # Otimizador Genético de escalação
│   └── analysis/                  # Análises estatísticas complementares
├── data/
│   ├── raw/                       # Dados brutos
│   ├── processed/                 # Resultados processados
│   └── models/                    # Modelos treinados salvos
├── main.py                        # Orquestrador principal do sistema
├── config.yaml                    # Configurações globais
└── requirements.txt               # Dependências do projeto
```

---

## 🚀 Funcionalidades

1. **📡 Coleta Resiliente**: Cliente API com retry automático e controle de taxa para evitar bloqueios.
2. **💾 Persistência Inteligente**: Banco de dados SQLite local para histórico completo de scouts.
3. **🤖 Predição Científica**: Modelos de Machine Learning treinados com validação temporal para estimar pontuações.
4. **🧬 Otimização Genética**: Algoritmo que encontra a escalação ideal respeitando formação tática e patrimônio.
5. **📊 Feature Engineering**: Médias móveis, tendências de momento e força de adversário integradas.

---

## 🛠️ Instalação e Uso

1. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar credenciais**:
   Crie um arquivo `.env` na raiz baseado no `.env.example`:
   ```env
   CARTOLA_EMAIL=seu@email.com
   CARTOLA_PASSWORD=sua_senha
   CARTOLA_PATRIMONIO=100.0
   CARTOLA_FORMACAO=4-3-3
   ```

3. **Executar o sistema**:
   ```bash
   python main.py
   ```

---

## 📊 Configurações

Ajuste os parâmetros de ML e Otimização diretamente no arquivo `config.yaml`.

- **Modelos**: Random Forest (padrão) ou Gradient Boosting.
- **Otimizador**: Tamanho da população, taxa de mutação e gerações configuráveis.

---

## 💡 Próximos Passos

- Integração com IA Perplexity para análise de notícias de última hora.
- Dashboard interativo via Streamlit.
- Backtesting de rodadas passadas.

---

**Bora mitar no Cartola com ciência de dados! 🏆⚽📊**
