# 🚀 Guia de Integração - Pipeline de Análise

> **Como usar o novo sistema de análise especialista antes de gerar a escalação**

---

## 🏆 Arquitetura do Pipeline

```
┌────────────────────┐
│  pipeline_escalacao.py        │
│  Orquestra todo o fluxo   │
└────────────────────┘
           ↑
           ↑ importa
           ↑
┌────────────────────┐
│  analise_especialista.py    │
│  Núcleo de análise (9 etapas) │
└────────────────────┘
           ↑
           ↑
      API Cartola FC
```

---

## 👯 Fluxo de Execução

### 1. Usar no `main.py` (Execução por CLI)

```python
from src.pipeline_escalacao import PipelineEscalacao

# Executar análise e gerar escala
pipeline = PipelineEscalacao(
    numero_rodada=7,
    estrategia="meio-termo"  # "mitar", "valorizar", "meio-termo"
)

# Executa pipeline completo
resultado = pipeline.executar_pipeline_completo(
    fazer_commit=True  # Auto-commit no Git
)

if resultado["sucesso"]:
    print(f"\nRelatório: {resultado['relatorio']}")
    print(f"Recomendação: {resultado['recomendacao']}")
```

**Saída:**
```
🚀 Iniciando análise especialista - Rodada 7...

[1/6] Mapeando confrontos...
  ✓ 10 confrontos classificados

[2/6] Analisando scouts...
  ✓ Scouts analisados para 7 posições

[3/6] Construindo time (meio-termo)...
  ✓ Time com 12 jogadores construído

[4/6] Selecionando capitão...
  ✓ Capitão: Pedro (Flamengo) - 28 pts

[5/6] Validando escalação...
  ✓ Escalação VÁLIDA

[6/6] Gerando relatório markdown...
  ✓ Relatório gerado (8500 chars)

============================================================
✅ ANÁLISE COMPLETA! Pronto para escalar.
============================================================
```

### 2. Usar no Streamlit (`app.py`)

```python
import streamlit as st
from src.pipeline_escalacao import PipelineEscalacao

st.set_page_config(page_title="Cartola - Análise Especialista")

# Sidebar com controles
st.sidebar.title("🎯 Cartola Especialista")
numero_rodada = st.sidebar.number_input("Rodada:", min_value=1, max_value=38, value=7)
estrategia = st.sidebar.selectbox(
    "Estratégia:",
    ["mitar", "valorizar", "meio-termo"]
)

if st.sidebar.button("🚀 Executar Análise"):
    with st.spinner("Analisando rodada..."):
        pipeline = PipelineEscalacao(numero_rodada, estrategia)
        resultado = pipeline.executar_pipeline_completo(fazer_commit=False)
    
    if resultado["sucesso"]:
        # Tab 1: Resumo
        tab1, tab2, tab3 = st.tabs(["Resumo", "Relatório", "JSON"])
        
        with tab1:
            st.success("✅ Análise Concluída!")
            pipeline.exibir_resumo_executivo()
        
        with tab2:
            st.markdown(pipeline.analisador.relatorio_md)
        
        with tab3:
            import json
            with open(resultado["recomendacao"]) as f:
                recomendacao = json.load(f)
            st.json(recomendacao)
    else:
        st.error(f"Erro: {resultado['erro']}")
```

---

## 📚 Estrutura de Dados

### Classe `Jogador`
```python
@dataclass
class Jogador:
    id: int                           # ID no Cartola
    nome: str                         # Nome do jogador
    time: str                         # Time (3 letras)
    posicao: str                      # "gol", "zagueiro", "lateral", etc
    preco: float                      # Preço em C$ (milhões)
    media_scouts: float               # Média de pontos/rodada
    media_por_90min: float            # Scouts normalizados por 90 min
    teto_pontos: float                # Máximo estimado esta rodada
    risco_cartao: str                 # "BAIXO", "MÉDIO", "ALTO"
    probabilidade_90min: float        # % chance de jogar 90 min (0-1)
    mpv: float                        # Mínimo Para Valorizar
    scouts_cedidos_oponente: float    # Pontos que oponente cede/rodada
    xG_xA: float                      # Expected Goals/Assists
```

### Classe `Confronto`
```python
@dataclass
class Confronto:
    id: int                          # ID da rodada
    time_a: str                      # Time A (mandante)
    time_b: str                      # Time B (visitante)
    mando_a: bool                    # True se A tem mando
    odds_vit_a: float                # Probabilidade de vitória de A (0-1)
    tipo: TipoConfrontoEnum           # GRUPO_A / GRUPO_B / GRUPO_C
    desfalques_a: List[str]          # Jogadores lesionados/suspensos
    desfalques_b: List[str]
    historico_gols: float            # Média histórica de gols
```

---

## 🔑 9 Etapas da Análise

| Etapa | Método | O que faz | Saída |
|-------|--------|----------|--------|
| **1** | `mapear_confrontos()` | Classifica cada jogo (Grupo A/B/C) | Dict com classificação |
| **2** | `analisar_scouts_por_posicao()` | Agrupa e ordena jogadores viáveis | Dict com Top 5 por posição |
| **3** | `calcular_mpv_jogadores()` | Calcula Mínimo Para Valorizar | Dict com MPV de cada jogador |
| **4** | `avaliar_matchup()` | Cruza jogador com oponente | (score: 0-1, justificativa) |
| **5** | `construir_time()` | Monta time 11+1 otimizado | List[Jogador] |
| **6** | `selecionar_capitao()` | Executa checklist de capitão | Jogador selecionado |
| **7** | `simular_cenarios()` | Simula 3 cenários | {"otimista": X, "realista": Y, "pessimista": Z} |
| **8** | `validar_escalacao()` | Valida erros/avisos | (válido, erros, avisos) |
| **9** | `gerar_relatorio_markdown()` | Cria relatório estruturado | String Markdown |

---

## 🌟 Checklist de Capitão

O capitão é selecionado automaticamente se atender TODOS ou MAIORIA destes critrios:

```python
checks = {
    "amplo_favoritismo": prob_vitoria > 0.65,  # Grupo A
    "minutos_90": probabilidade_90_min > 0.85,  # Confirmado titular
    "responsavel_bola_parada": eh_penaltista or eh_cobrador_falta,
    "media_recente_alta": media_scouts > media_geral * 1.2,
    "matchup_favoravel": scouts_cedidos > media_por_90min,
}

# Minímo 4/5 checks para ser capitão
if sum(checks.values()) >= 4:
    eh_capitao = True
```

---

## 📁 Saídas Geradas

### Arquivo 1: Relatório Markdown
**Path:** `docs/RODADA_7_ANALISE_20260318_1450.md`

Contains:
- Resumo executivo
- Mapeamento de confrontos (tabela)
- Análise de desfalques
- Scouts por posição (tabelas)
- Time montado (estrutura)
- Capitão recomendado
- 3 cenários simulados
- Validação pré-escala

### Arquivo 2: Recomendação JSON
**Path:** `output/RODADA_7_RECOMENDACAO.json`

```json
{
  "rodada": 7,
  "data_analise": "2026-03-18T14:50:00",
  "estrategia": "meio-termo",
  "time_recomendado": [
    {
      "nome": "Pedro",
      "time": "FLA",
      "posicao": "atacante",
      "preco": 19.50,
      "media_scouts": 16.5,
      "teto_pontos": 28
    },
    ...
  ],
  "capitao": {
    "nome": "Pedro",
    "time": "FLA",
    "teto_pontos": 28
  },
  "validacao": {
    "valido": true,
    "erros": [],
    "avisos": []
  }
}
```

---

## 💬 Commit Automático

Após gerar relatório e recomendação, o pipeline faz commit automático:

```bash
git add docs/RODADA_7_ANALISE_*.md output/RODADA_7_RECOMENDACAO.json
git commit -m "docs: add analysis and recommendation for round 7 (meio-termo)"
git push origin main
```

**Desabilitar commit:**
```python
pipeline.executar_pipeline_completo(fazer_commit=False)
```

---

## 🚀 Exemplos de Uso

### Exemplo 1: Executar do terminal
```bash
python -m src.pipeline_escalacao
```

### Exemplo 2: Executar do Streamlit
```bash
streamlit run app.py
```

### Exemplo 3: Integrar no main.py
```python
if __name__ == "__main__":
    from src.pipeline_escalacao import PipelineEscalacao
    
    # Rodada 7 com estratégia "meio-termo"
    pipeline = PipelineEscalacao(7, "meio-termo")
    pipeline.executar_pipeline_completo()
```

---

## ⚠️ Limitações Atuais

- [ ] API Cartola ainda não integrada (usar mock)
- [ ] xG/xA requer integração com modelos exter nos
- [ ] Heat maps posicionais ainda não impl ementados
- [ ] Perfilagem de árbitros ainda não implementada

---

## 💫 Próximas Melhorias

- [ ] Integrar API Cartola FC (https://api.cartolafc.globo.com)
- [ ] Treinar modelo XGBoost para predição de pontos
- [ ] Dashboard em tempo real com Streamlit
- [ ] Not ificações por WhatsApp/Telegram
- [ ] Banco de dados com histórico de rodadas
- [ ] Relatório automático por email

---

**Este sistema foi criado para eliminar dúvidas e sistematizar a escalação em Cartola. Use com sabedoria! 🚀**
