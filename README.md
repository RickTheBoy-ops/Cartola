<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="ML Badge"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite Badge"/>
</div>

<h1 align="center">🏆 Cartola FC - Ultimate Optimizer (AI-Powered)</h1>

<p align="center">
  <em>Sistema profissional de análise, predição e otimização para Cartola FC utilizando Machine Learning e Algoritmos Genéticos.</em>
</p>

<hr>

## 🚀 Sobre o Projeto

O **Cartola FC - Ultimate Optimizer** é uma ferramenta definitiva para quem deseja mitar no Cartola FC usando ciência de dados. Integrado com **Machine Learning** avançado e **Otimização Genética**, ele não apenas sugere jogadores baseados no histórico, mas estima pontuações futuras e encontra a escalação matematicamente ideal para maximizar seus pontos respeitando seu patrimônio em cartoletas.

---

## ✨ Funcionalidades Principais

| Recurso | Descrição |
| :--- | :--- |
| **📡 Coleta Resiliente** | Cliente de API com retry automático e rate limiting para extração dos scouts oficiais sem bloqueios. |
| **💾 Persistência Inteligente**| Banco SQLite local ultrarrápido mantendo o histórico de todos os scouts para análises longas. |
| **🤖 Predição Científica** | Utilizando Random Forest e Gradient Boosting treinado com validação temporal da performance. |
| **🧬 Otimização Genética** | Algoritmo que gera milhões de cruzamentos para achar a equipe ideal (formação tática x precificação). |
| **📊 Feature Engineering** | Cálculos de médias móveis, tendências de momento, e força do adversário integradas. |

---

## 📁 Estrutura do Repositório

```text
cartola/
├── src/
│   ├── api/          # 🌐 Cliente HTTP robusto
│   ├── data/         # 🗄️ Pipeline ETL e persistência (SQLite)
│   ├── ml/           # 🧠 Modelos ML e Feature Engineering avançado
│   └── analysis/     # 📈 Análises estatísticas complementares
├── data/
│   ├── raw/          # Dados originais não tratados
│   ├── processed/    # Dados finais formatados
│   └── models/       # Modelos salvos (arquivos picke)
├── main.py           # 🎯 Orquestrador unificado
├── config.yaml       # ⚙️ Configurações globais e táticas
└── requirements.txt  # 📦 Dependências do ambiente
```

---

## 🛠️ Guia de Instalação e Uso

### 1️⃣ Pré-requisitos
Certifique-se de ter o Python instalado. Clone este repositório e baixe as dependências:
```bash
git clone https://github.com/RickTheBoy-ops/Cartola.git
cd Cartola
pip install -r requirements.txt
```

### 2️⃣ Configuração do Ambiente (.env)
Você precisará de um arquivo `.env` contendo suas variáveis e estratégia inicial. Crie com base no `.env.example`:

```env
CARTOLA_EMAIL=seu@email.com
CARTOLA_PASSWORD=sua_senha
CARTOLA_PATRIMONIO=100.0
CARTOLA_FORMACAO=4-3-3
```

### 3️⃣ Configuração Fina (config.yaml)
Se preferir, ajuste o algoritmo pelo arquivo `config.yaml`:
- **Modelos**: Escolha entre `Random Forest` ou `Gradient Boosting`.
- **Otimizador Genético**: Configure número de gerações, taxa de mutação e população.

### 4️⃣ Executando
Para rodar nosso orquestrador completo (coleta -> feature eng -> treino -> escalação heurística):
```bash
python main.py
```

---

## 💡 Próximos Passos do RoadMap
- [ ] 🤖 Integração com IA Perplexity (análise de notícias de última hora)
- [ ] 📊 Dashboard Interativo (Streamlit) para visualização do time
- [ ] 🕰️ Backtesting Automático simulando ganhos em rodadas anteriores

---

<div align="center">
  <b>Bora mitar no Cartola com ciência de dados! 🏆⚽📊</b>
  <br>
  <sub>Sinta-se livre para contribuir, avaliar e dar uma ⭐ no projeto.</sub>
</div>
