# ⚽ Cartola FC - Sistema V2 (Refatorado)

> **Nova arquitetura modular com Programação Linear, Feature Engineering avançado e Factory Pattern**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PuLP](https://img.shields.io/badge/PuLP-2.7%2B-green)](https://coin-or.github.io/pulp/)
[![Status](https://img.shields.io/badge/Status-Refactored-success)](https://github.com/RickTheBoy-ops/Cartola)

---

## 🎯 O que mudou?

### ❌ **Sistema Antigo**
- Arquivo monolítico (`cartola_mega_optimizer.py` com 800+ linhas)
- Feature Engineering e otimização acoplados
- Difícil de testar e estender
- Sem separação de responsabilidades

### ✅ **Sistema V2 (Atual)**
- **Modular**: Código organizado em pacotes distintos
- **Extensível**: Factory Pattern para novas estratégias
- **Testável**: Cada módulo independente
- **Manutenibilidade**: Código limpo com docstrings
- **Performance**: Otimização aprimorada

---

## 📊 Nova Estrutura

```
src/
├── optimizer/
│   ├── base.py              # Interface base (ABC)
│   ├── mega_strategy.py     # PuLP optimizer
│   ├── factory.py           # Factory Pattern
│   └── __init__.py
├── features/
│   ├── feature_engineering_v2.py  # Análise por posição
│   └── __init__.py
├── utils/
│   └── __init__.py
└── __init__.py

examples/
└── optimize_lineup.py   # Exemplo completo

MIGRATION_GUIDE.md       # Guia de migração detalhado
README_V2.md             # Este arquivo
```

---

## 🚀 Quick Start

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Uso básico

```python
import pandas as pd
from optimizer import CartolaOptimizer
from features import FeatureEngineeringV2

# Carregar dados
df_raw = pd.read_csv('jogadores.csv')

# Feature Engineering
feature_eng = FeatureEngineeringV2()
df_enriched = feature_eng.engineer_features(df_raw)

# Otimizar
optimizer = CartolaOptimizer(strategy='mega')
lineup = optimizer.optimize(df_enriched, budget=100.0)

print(lineup[['nome', 'posicao_id', 'mega_score', 'preco']])
```

---

## 📋 Principais Módulos

### 1️⃣ **OptimizerStrategy (Base)**

Classe abstrata que define interface para otimizadores:

```python
from optimizer.base import OptimizerStrategy

class MyStrategy(OptimizerStrategy):
    name = "My Custom Strategy"
    
    def optimize(self, df, budget, formation=None, **kwargs):
        # Lógica de otimização
        return lineup
```

**Métodos obrigatórios:**
- `optimize()` - Retorna escalação otimizada
- `validate()` - Valida escalação
- `select_captain()` - Seleciona capitão

### 2️⃣ **MegaStrategy (PuLP)**

Otimização usando Programação Linear Inteira:

**Características:**
- ✅ Solução matematicamente ótima
- ✅ Testa todas as formações (7 opções)
- ✅ Restrições: formação, orçamento, clubes
- ✅ Conflitos de adversários (DEF vs ATK)

**Configuração:**

```python
optimizer = CartolaOptimizer(
    strategy='mega',
    config={
        'max_players_per_club': 3,
        'enable_opponent_conflicts': True,
        'solver_time_limit': 30,
        'test_all_formations': True
    }
)
```

### 3️⃣ **FeatureEngineeringV2**

Análise estatística POR POSIÇÃO:

**Novidades:**
- ✅ **Z-score por posição**: GOL, LAT, ZAG, MEI, ATA têm estatísticas próprias
- ✅ **Percentis**: Identifica jogadores top em cada posição
- ✅ **Detecção de outliers**: Remove valores extremos
- ✅ **Pesos dinâmicos**: Critérios diferentes por posição

**Uso:**

```python
feature_eng = FeatureEngineeringV2(config={
    'min_games': 3,
    'use_percentiles': True,
    'use_zscore_normalization': True,
    'remove_outliers': True,
    'outlier_threshold': 3.0
})

df_enriched = feature_eng.engineer_features(df_raw)

# Top jogadores
top_10 = feature_eng.get_top_players(df_enriched, n=10)

# Estatísticas
stats = feature_eng.get_position_stats(df_enriched)
```

**Pesos por posição (padrão):**

| Posição | Média | Últimas 5 | Variância | Jogos | Minutos |
|---------|--------|------------|-----------|-------|--------|
| **GOL** | 0.40 | 0.30 | -0.15 | 0.10 | 0.15 |
| **LAT** | 0.35 | 0.35 | -0.10 | 0.10 | 0.10 |
| **ZAG** | 0.40 | 0.30 | -0.15 | 0.10 | 0.15 |
| **MEI** | 0.30 | 0.40 | -0.05 | 0.10 | 0.15 |
| **ATA** | 0.30 | 0.45 | -0.05 | 0.05 | 0.15 |

### 4️⃣ **CartolaOptimizer (Factory)**

Factory Pattern para criar otimizadores:

```python
# Usar estratégia existente
optimizer = CartolaOptimizer(strategy='mega')

# Registrar nova estratégia
CartolaOptimizer.register_strategy('custom', MyCustomStrategy)

# Usar nova estratégia
optimizer = CartolaOptimizer(strategy='custom')
```

**Métodos disponíveis:**

```python
# Otimizar
lineup = optimizer.optimize(df, budget=100)

# Validar
is_valid = optimizer.validate(lineup, budget=100, formation='3-4-3')

# Capitão
captain = optimizer.select_captain(lineup)

# Info
info = optimizer.get_info()
strategies = optimizer.get_available_strategies()
```

---

## 🔧 Comparativo: Antes vs Agora

### **Antes (Sistema Antigo)**

```python
from cartola_mega_optimizer import cartola_mega_optimizer

df = pd.read_csv('jogadores.csv')
lineup = cartola_mega_optimizer(df, budget=100, formation='3-4-3')
```

**Problemas:**
- Tudo em um único arquivo
- Impossível testar isoladamente
- Difícil adicionar novas estratégias
- Features misturadas com otimização

### **Agora (Sistema V2)**

```python
from optimizer import CartolaOptimizer
from features import FeatureEngineeringV2

# Separar responsabilidades
feature_eng = FeatureEngineeringV2()
df_enriched = feature_eng.engineer_features(df_raw)

optimizer = CartolaOptimizer(strategy='mega')
lineup = optimizer.optimize(df_enriched, budget=100)
```

**Benefícios:**
- ✅ Modular e organizado
- ✅ Fácil de testar
- ✅ Extensível (Factory Pattern)
- ✅ Separação clara de responsabilidades

---

## 📚 Documentação Completa

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guia completo de migração
- **[examples/optimize_lineup.py](examples/optimize_lineup.py)** - Exemplo funcional
- **Docstrings** - Todos os módulos documentados

---

## 🧠 Exemplo Completo

Veja [examples/optimize_lineup.py](examples/optimize_lineup.py) para exemplo completo.

**Output esperado:**

```
======================================================================
CARTOLA FC - SISTEMA DE OTIMIZAÇÃO V2
======================================================================

📊 Carregando dados...
   ✅ 50 jogadores carregados

🧠 Executando Feature Engineering V2...
   ✅ Features criadas! Mega_score médio: 52.34

🎯 Top 10 Jogadores por Mega Score:
   Jogador_15      | Pos: 1 | Mega: 87.3 | Preço: C$25.0
   Jogador_42      | Pos: 5 | Mega: 84.1 | Preço: C$32.0
   ...

👥 Estatísticas por Posição:
   GOL: 10 jogadores | Média: 48.2
   LAT: 10 jogadores | Média: 51.7
   ZAG: 10 jogadores | Média: 49.3
   MEI: 10 jogadores | Média: 53.8
   ATA: 10 jogadores | Média: 56.4

🚀 Iniciando otimização...
🎯 Otimizador inicializado: Mega Strategy (PuLP)

🧠 Testando 7 formações...
   ⚽ 3-4-3: Score=645.2 | Custo=C$99.5
   ⚽ 3-5-2: Score=632.1 | Custo=C$98.2
   ⚽ 4-3-3: Score=638.7 | Custo=C$99.8
   ⚽ 4-4-2: Score=628.9 | Custo=C$97.5
   ⚽ 4-5-1: Score=621.3 | Custo=C$96.8
   ⚽ 5-3-2: Score=635.4 | Custo=C$98.9
   ⚽ 5-4-1: Score=618.7 | Custo=C$95.2

   🏆 MELHOR FORMAÇÃO: 3-4-3 (Score: 645.2)

======================================================================
🏆 ESCALAÇÃO OTIMIZADA
======================================================================

💰 Custo Total: C$ 99.50 / C$ 100.00
🎯 Score Total: 645.20
⚽ Jogadores: 12

👥 Lineup:
   Jogador_15      | Pos: 1 | Mega:  87.3 | Média: 12.5 | Preço: C$ 25.0
   Jogador_23      | Pos: 2 | Mega:  73.1 | Média:  9.8 | Preço: C$ 18.0
   ...

🎺 Capitão: Jogador_15

======================================================================
```

---

## 🛣️ Roadmap

### V2.1 (Próximas Entregas)

- [ ] **GeneticStrategy** - Algoritmo Genético
- [ ] **EnsembleStrategy** - Combinação de múltiplas estratégias
- [ ] **Testes Unitários** - Cobertura completa
- [ ] **CI/CD** - GitHub Actions

### V2.2 (Planejado)

- [ ] API REST
- [ ] Dashboard Streamlit
- [ ] Integração Cartola API
- [ ] Docker support

---

## ❓ FAQ

**1. Preciso migrar tudo de uma vez?**

Não! A migração é incremental. Consulte [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

**2. O código antigo ainda funciona?**

Sim, mas recomendamos migrar para aproveitar melhorias.

**3. Como adicionar nova estratégia?**

Use Factory Pattern:

```python
class MyStrategy(OptimizerStrategy):
    # implementação
    pass

CartolaOptimizer.register_strategy('my_strategy', MyStrategy)
```

**4. Qual a diferença no desempenho?**

Performance similar ou melhor. Principal ganho é manutenibilidade.

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit (`git commit -m 'Add: nova feature'`)
4. Push (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📞 Contato

**Autor:** Erick Vinicius  
**GitHub:** [@RickTheBoy-ops](https://github.com/RickTheBoy-ops)

---

**⚽ Boa sorte no Cartola FC! 🏆**
