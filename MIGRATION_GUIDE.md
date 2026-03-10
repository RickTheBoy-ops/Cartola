# Guia de Migração - Cartola FC V2

## 📚 Visão Geral

Este guia ajuda a migrar do código antigo para o novo sistema refatorado.

---

## 🔄 Comparação: Antes vs Agora

### **Sistema Antigo**

```python
# Arquivo monolítico: cartola_mega_optimizer.py

import pandas as pd
from cartola_mega_optimizer import cartola_mega_optimizer

df = pd.read_csv('jogadores.csv')
lineup = cartola_mega_optimizer(df, budget=100, formation='3-4-3')
```

**Problemas:**
- ❌ Arquivo único com 800+ linhas
- ❌ Acoplamento alto (features + optimizer juntos)
- ❌ Difícil testar e estender
- ❌ Sem separação de responsabilidades

---

### **Sistema Novo (V2)**

```python
# Estrutura modular

import pandas as pd
from optimizer import CartolaOptimizer
from features import FeatureEngineeringV2

# 1. Feature Engineering
feature_eng = FeatureEngineeringV2()
df_enriched = feature_eng.engineer_features(df_raw)

# 2. Otimização
optimizer = CartolaOptimizer(strategy='mega')
lineup = optimizer.optimize(df_enriched, budget=100)
```

**Benefícios:**
- ✅ Modular e organizado
- ✅ Fácil de testar
- ✅ Extensível (novas estratégias)
- ✅ Separação clara de responsabilidades

---

## 🛠️ Passos de Migração

### **Passo 1: Instalar Dependências**

```bash
pip install -r requirements.txt
```

**Dependências principais:**
- `pulp` - Programação Linear
- `pandas` - Manipulação de dados
- `numpy` - Operações numéricas

---

### **Passo 2: Adaptar Código de Carregamento**

#### Antes:
```python
df = pd.read_csv('jogadores.csv')
lineup = cartola_mega_optimizer(df, budget=100)
```

#### Agora:
```python
from optimizer import CartolaOptimizer
from features import FeatureEngineeringV2

# Carregar dados brutos
df_raw = pd.read_csv('jogadores.csv')

# Enriquecer com features
feature_eng = FeatureEngineeringV2()
df_enriched = feature_eng.engineer_features(df_raw)

# Otimizar
optimizer = CartolaOptimizer(strategy='mega')
lineup = optimizer.optimize(df_enriched, budget=100)
```

---

### **Passo 3: Configurações Personalizadas**

#### Antes:
```python
# Sem configurações flexíveis
lineup = cartola_mega_optimizer(df, budget=100, formation='3-4-3')
```

#### Agora:
```python
# Configurações via dicionário
optimizer = CartolaOptimizer(
    strategy='mega',
    config={
        'max_players_per_club': 3,
        'enable_opponent_conflicts': True,
        'solver_time_limit': 30,
        'test_all_formations': True
    }
)

lineup = optimizer.optimize(df_enriched, budget=100)
```

---

### **Passo 4: Testar Novas Features**

#### Análise por Posição:

```python
feature_eng = FeatureEngineeringV2()
df_enriched = feature_eng.engineer_features(df_raw)

# Estatísticas por posição
stats = feature_eng.get_position_stats(df_enriched)
print(stats)

# Top jogadores
top_10 = feature_eng.get_top_players(df_enriched, n=10)
print(top_10[['nome', 'posicao_id', 'mega_score', 'preco']])
```

---

## 📊 Features Novas (V2)

### **1. Feature Engineering V2**

**Melhorias:**
- ✅ Normalização por posição (Z-score)
- ✅ Percentis por posição
- ✅ Detecção de outliers
- ✅ Pesos dinâmicos por posição

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
```

---

### **2. Factory Pattern (Extensível)**

**Adicionar nova estratégia:**

```python
from optimizer.base import OptimizerStrategy
from optimizer import CartolaOptimizer

class MyCustomStrategy(OptimizerStrategy):
    name = "Custom Greedy"
    
    def optimize(self, df, budget, formation=None, **kwargs):
        # Lógica customizada
        return lineup

# Registrar nova estratégia
CartolaOptimizer.register_strategy('custom', MyCustomStrategy)

# Usar
optimizer = CartolaOptimizer(strategy='custom')
lineup = optimizer.optimize(df, budget=100)
```

---

### **3. Validação Automática**

```python
lineup = optimizer.optimize(df_enriched, budget=100)

# Validar escalação
is_valid = optimizer.validate(lineup, budget=100, formation='3-4-3')
print(f"Escalação válida: {is_valid}")

# Selecionar capitão
captain = optimizer.select_captain(lineup)
print(f"Capitão: {captain}")
```

---

## 📝 Checklist de Migração

- [ ] Instalar dependências (`pip install -r requirements.txt`)
- [ ] Atualizar imports para novo sistema modular
- [ ] Adicionar Feature Engineering V2 antes da otimização
- [ ] Substituir chamadas diretas por `CartolaOptimizer`
- [ ] Testar escalações com validação
- [ ] Ajustar configurações personalizadas
- [ ] Remover imports antigos (`cartola_mega_optimizer`)

---

## 🎯 Comparação de Performance

| Métrica | Sistema Antigo | Sistema V2 |
|---------|----------------|------------|
| **Linhas de código** | 800+ | ~300 por módulo |
| **Modularidade** | Nenhuma | Alta |
| **Testabilidade** | Baixa | Alta |
| **Extensibilidade** | Difícil | Fácil (Factory) |
| **Manutenção** | Difícil | Fácil |
| **Performance** | Boa | Igual ou melhor |

---

## ❓ FAQ

### **1. O código antigo ainda funciona?**

Sim, mas recomendamos migrar para o V2 para aproveitar melhorias.

### **2. Preciso reescrever tudo?**

Não! A migração é incremental:
1. Instalar dependências
2. Atualizar imports
3. Adicionar Feature Engineering V2
4. Substituir chamadas antigas

### **3. Como testar sem quebrar meu código?**

Crie branch separada:
```bash
git checkout -b feature/migrate-to-v2
```

### **4. Qual a diferença entre 'mega' e outras estratégias?**

Atualmente, apenas 'mega' está implementada (Programação Linear).

Futuramente:
- `genetic`: Algoritmo Genético
- `ensemble`: Combinação de múltiplas estratégias

### **5. Como contribuir com novas estratégias?**

1. Herdar de `OptimizerStrategy`
2. Implementar método `optimize()`
3. Registrar via `CartolaOptimizer.register_strategy()`

Veja exemplo em `MIGRATION_GUIDE.md` (seção Factory Pattern).

---

## 📞 Suporte

Problemas na migração?

1. Consulte `examples/optimize_lineup.py`
2. Verifique `README.md` principal
3. Abra issue no GitHub

---

**✅ Migração concluída! Aproveite o novo sistema V2!**
