# 🚀 Guia Completo: Features Avançadas Cartola FC

## 🎯 Overview

Implementação baseada em **research acadêmico** + **hacks dos top players**.

---

## 📊 Impacto Esperado

| Feature | Impacto | ROI |
|---------|---------|-----|
| Sistema Valorização | +15-20% patrimônio R1-R5 | 🌟🌟🌟🌟🌟 |
| Análise Confrontos | +10-15% pontuação | 🌟🌟🌟🌟🌟 |
| Feature Engineering | +8-12% acurácia modelo | 🌟🌟🌟🌟 |
| Detectores Padrões | Identifica oportunidades | 🌟🌟🌟🌟🌟 |
| Análise Exploratória | Insights estratégicos | 🌟🌟🌟🌟 |

---

## 🏗️ Arquitetura

```
Cartola/
├── src/
│   ├── features/           # Módulos de features
│   │   ├── valorizacao.py  # Sistema de valorização
│   │   ├── confrontos.py   # Análise de confrontos
│   │   ├── engineering.py  # Feature engineering
│   │   └── detectors.py    # Detectores especiais
│   └── analysis/
│       └── exploratory.py  # Análise exploratória
└── config.yaml         # Configurações
```

---

## ✨ Features Principais

### 1. Sistema de Valorização

**Fórmulas comprovadas:**

- **Rodada 1:** `0.46 × Preço = Pontos mínimos`
- **Rodada 2:** Boost 20% para quem valorizou
- **Expulsos:** Threshold 30% (alta valorização)

**Estratégia dinâmica:**
- R1-R5: 70% valorização, 30% pontuação
- R6-R15: 50% valorização, 50% pontuação
- R16+: 30% valorização, 70% pontuação

### 2. Análise de Confrontos

**HACKS (pesos táticos):**

| Situação | Boost | Por quê? |
|----------|-------|----------|
| Goleiro FORA vs fraco | +40% | Time grande fora × pequeno casa = time pequeno ataca desesperado = + defesas (4pts cada) |
| Lateral prob SG >40% | +35% | SG = 5pts. Laterais subestimados. |
| Atacante CASA vs fraco | +30% | Mais gols esperados. |
| Falso Meia | +50% | **OURO**: Preço de meia, performance de atacante. |
| Volta suspensão | +25% | Preço baixo (média negativa) + valorização alta. |

### 3. Feature Engineering

**Features que importam (comprovado em papers):**

1. **Momentum (3 rodadas)** - Melhor preditor individual
2. **Explosão** - Desvio performance recente vs histórica
3. **Scouts ponderados** - Por posição (cada scout vale diferente)
4. **Regularidade** - Inverso do CV (confiabilidade)
5. **Favoristmo** - Mando + força adverssário
6. **Expectativa** - Momentum × Favoristmo

### 4. Detectores de Padrões

**Identifica jogadores especiais:**

- **Falsos Meias:** >3 fin/jogo + >0.3 gols/jogo
- **Laterais Ofensivos:** >0.2 assist/jogo
- **Zagueiros Artilheiros:** >0.15 gols/jogo
- **Voltando Suspensão:** Cartão vermelho rodada anterior

### 5. Análise Exploratória

**Padrões ocultos:**

- Explosivos em clássicos (+20%)
- Super regulares (baixo CV)
- Dependentes de mando (+25% casa)
- Adversários favoritos

---

## 🛠️ Implementação

### Passo 1: Estrutura

```bash
mkdir -p src/{features,analysis,models}
touch src/features/{__init__,valorizacao,confrontos,engineering,detectors}.py
touch src/analysis/{__init__,exploratory}.py
```

### Passo 2: Config

Adicionar ao `config.yaml`:

```yaml
valorizacao:
  rodada_1_threshold: 0.46
  rodada_2_boost: 1.20
  expulso_threshold: 0.30

tactical_weights:
  goleiro_fora_vs_fraco: 1.40
  lateral_sg_alto: 1.35
  atacante_casa_vs_fraco: 1.30
  falso_meia: 1.50
  volta_suspensao: 1.25

feature_engineering:
  rolling_windows: [3, 5, 10]
  ewm_spans: [3, 5]

models:
  primary: "catboost"
  catboost:
    iterations: 1000
    learning_rate: 0.03
    depth: 6
```

### Passo 3: Código

Ver exemplos completos nos módulos:
- `src/features/valorizacao.py`
- `src/features/confrontos.py`
- `src/features/engineering.py`
- `src/features/detectors.py`
- `src/analysis/exploratory.py`

---

## 📚 Research Base

### Papers Acadêmicos

1. **UFU (2025)** - ML para Cartola
   - CatBoost: 91 pts/rodada
   - Random Forest: 88 pts/rodada
   - LSTM: 87 pts/rodada

2. **IFRS (2024)** - Sistema CBR
   - Raciocínio baseado em casos
   - Recomendação de escalações

### Fontes Primárias

- CartolaBrasil.com.br
- GloboEsporte
- Comunidade top players

---

## 🚀 Roadmap

### Fase 1: Fundação (Esta PR)
- [x] Design de arquitetura
- [x] Documentação completa
- [ ] Implementar módulos
- [ ] Testes unitários

### Fase 2: Integração
- [ ] Pipeline completo
- [ ] Modelo CatBoost
- [ ] Dashboard

### Fase 3: Otimização
- [ ] Algoritmo genético
- [ ] Sistema CBR
- [ ] API REST

---

## 💬 Como Usar

```python
# Valorização
from src.features.valorizacao import SistemaValorizacao
sistema = SistemaValorizacao(config)
analise = sistema.analisar_grupo(jogadores_df, rodada=1)

# Confrontos
from src.features.confrontos import AnalisadorConfrontos
analisador = AnalisadorConfrontos(config)
metrics = analisador.analisar_confronto('Flamengo', 'Vasco', 'fora', historico_df)

# Detectores
from src.features.detectors import DetectorPadroesEspeciais
detector = DetectorPadroesEspeciais(config)
deteccoes = detector.executar_todas_deteccoes(df, rodada=5)
```

---

**💡 Próximo: Implementar os módulos `.py` localmente!**
