# 🧠 HEURÍSTICAS DE IA E PADRÕES AVANÇADOS

> **Guia para integração de Machine Learning, heurísticas e modelagem estatística no projeto Cartola**

---

## 1︠︡︠ NORMALIZAÇÕES E MÉTRICAS AVANÇADAS

### Média por 90 Minutos (Métrica Chave)

**Problema:** Jogador entra aos 70′ com 5 pontos. Sua média = 5 pontos, mas não teve tempo de atuar.

**Solução:** Normalizar por tempo:

```python
pts_por_90_min = (pontos_total / minutos_jogados) * 90

# Exemplo:
jogador_entrou_30_min = 3_pontos
jogador_30_90 = (3 / 30) * 90 = 9 pontos/90min
```

**Uso:**
- Comparação equitativa entre suplentes e titúlares
- Identifica "azarados" que explodem quando ganham tempo
- Melhor preditor que média bruta

---

### xG (Expected Goals) e xA (Expected Assists)

**Definição:** Modelos de machine learning calculam a "qualidade" das chances criadas/perdidas.

```
xG = 0.8 → 80% de probabilidade de ter resultado em gol
xA = 1.2 → 1.2 assist(s) esperado(s) pela qualidade das passes
```

**Implicado:**
- Um atacante com 0 gols mas xG = 2.3 está **muito azarado** → bombar na próxima
- Um atacante com 2 gols mas xG = 0.5 está **muito feliz** → pode normalizar

**Implementação:**
1. Coletar dados de cada finalização:
   - Distância do gol
   - Ângulo
   - Tipo de finalização (cabeça, pé, domina)
   - Defesa oferecer (marcado ou livre)
2. Usar modelo Logit/LightGBM para estimar probab.
3. Acumular xG por jogador/rodada

---

### Scouts Cedidos por Posição

**Problema:** "Time X cede 7 pontos/rodada para laterais". Muito vago.

**Solução:** Granularidade:

```python
# Por posição X lado X mando
scouts_cedidos[
    time_oponente = "Flamengo",
    posicao = "LAT-ESQ",
    mando = "FORA"
] = 8.2  # Flamengo cede 8.2 pts para LAT esquerdos quando joga fora

# Usar para predição
scout_esperado_lateral = scouts_cedidos[time_oponente][posicao][mando]
```

**Impacto:** Transforma "achismo" em dados concretos.

---

## 2︠︡︠ RECLASSIFICAÇÃO TÁTICA (Posicional Real)

### O Problema

No Cartola, um "lateral" é registrado como LAT (6 pontos por gol) mas que atua como **ponta ofensiva** (8 pontos).

### A Solução

**Mapa de Calor / Heat Map:**
1. Coletar posições do jogador no campo a cada toque (do transmissor/ESPN)
2. Gerar mapa de densidade
3. Se concentração > 60% em zona ofensiva (ex: últa terço ofensivo) → reclassificar como **"Ponta"**

**Benefício na Escalação:**
- Um lateral que atua como ponta merece teto de pontos maior
- Alterar seu multiplicador de capitão ou seu scout base

**Implementação:**
```python
class JogadorAI:
    posicao_oficial = "LAT"  # Cartola
    posicao_real = "PONTA"  # Mapa de calor
    multiplicador_capitao = 2.0 if posicao_real == "PONTA" else 1.5
    scout_esperado = scout_por_posicao[posicao_real]  # Usar real, não oficial
```

---

## 3︠︡︠ CLUSTERIZAÇÃO DE RODADAS

### Classificação Automática dos Jogos

Usando K-Means com variáveis:
- Diferença de ranking (ELO ou pontos)
- Odds de vitória (mercado)
- Histórico de gols no confronto
- Posicional de cada time (defesa forte vs ofensa forte)

```python
# Saída esperada:
grupo_rodada = {
    "Grupo A (Amplo Favoritismo)": [
        {"time_fav": "Flamengo", "time_und": "Cuiabá", "probs": {"vit": 0.72, "sg": 0.18, "2+": 0.65}},
        ...
    ],
    "Grupo B (Aberto/Gols ambos)": [
        {"time1": "Corinthians", "time2": "São Paulo", "probs": {"2+": 0.78, "3+": 0.52}},
        ...
    ],
    "Grupo C (Truncado/Defesa)": [
        {"time1": "Botafogo", "time2": "Vasco", "probs": {"sg": 0.35, "0x0": 0.20}},
        ...
    ]
}
```

**Regras por Grupo:**

| Grupo | Estratégia | Exemplos de Escalação |
|-------|-----------|-------------------------|
| **A** | Garantir defesa + 2 ofensivos | Gol de time favorito + 1 defensor de ambos |
| **B** | Atacantes + Meias ofensivas | Ambos atacantes + 2 meias ofensivas |
| **C** | Fugir | Nenhum escalado ou apenas 1 volante rebatedor |

---

## 4︠︡︠ PERFILAGEM DE ÁRBITROS

### Padrão de Cartões

**Implementação:**

```python
arbitro_perfil = {
    "nome": "Braulio da Silva",
    "amarelos_por_jogo": 5.2,  # Média histórica
    "vermelhos_por_jogo": 0.3,
    "faltas_toleradas": 18,  # Antes de amarelo
    "tendencia": "Complacente",  # vs Punitivo
}

# Usar para evitar escalando jogadores "infratores"
if jogador.cartoes_acumulados > 3 and jogo.arbitro.tendencia == "Punitivo":
    risco_cartao = "ALTO"
else:
    risco_cartao = "MÉDIO"
```

---

## 5︠︡︠ LIMITE DE CORRELAÇÃO (Risco Geométrico)

### Problema: "Morrer Abraçado"

Escalar 5 defensores do mesmo time que leva 1 gol = -5 defesores pontuam e o time sofre 1 gol = -3 defesores adicionais.

### Solução: Limite Automático

```python
def validar_correlacao(time_selecionado, meu_time):
    defensores_mesmo_time = sum(
        1 for j in meu_time 
        if j.time == time_selecionado and j.posicao in ["GOL", "ZAG", "LAT"]
    )
    
    if defensores_mesmo_time > 3:
        return False, f"Máximo 3 defensores do mesmo time. Você tem {defensores_mesmo_time}."
    
    atacantes_mesmo_time = sum(
        1 for j in meu_time 
        if j.time == time_selecionado and j.posicao == "ATA"
    )
    
    if atacantes_mesmo_time > 2:
        return False, f"Máximo 2 atacantes do mesmo time. Você tem {atacantes_mesmo_time}."
    
    return True, "Ok"
```

---

## 6︠︡︠ CÁLCULO DO CAPITÃO (Algoritmo Determinista)

### Checklist Automático

```python
def candidato_capitao_valido(jogador, rodada):
    checks = {
        "amplo_favoritismo": jogador.time.prob_vitoria > 0.65,  # Grupo A
        "minutos_90": jogador.probabilidade_90_min > 0.85,
        "responsavel_bola_parada": jogador.eh_penaltista or jogador.eh_cobrador_falta,
        "media_recente_alta": jogador.media_ultimas_5 > jogador.media_geral * 1.2,
        "matchup_favoravel": (
            scout_cedido[jogador.posicao][oponente][mando] > 
            jogador.media_por_90_min
        ),
    }
    
    pontuacao_capitao = sum(checks.values())  # 0-5
    
    if pontuacao_capitao >= 4:
        return True  # Candidato forte
    else:
        return False  # Espere próxima rodada
```

**Resultado:** Lista de 2-3 candidatos ordenados por "força" da candidatura.

---

## 7︠︡︠ PREVISÃO DE PONTUAÇÃO (Modelo Completo)

### Variáveis de Entrada

```python
def prever_pontos_jogador(jogador, rodada_dados):
    features = [
        jogador.media_por_90_min,
        jogador.desvio_padrao_pontos,  # Consistência
        scout_cedido[jogador.posicao][oponente][mando],
        oponente.poder_ofensivo if jogador.eh_defensor else oponente.poder_defensivo,
        jogador.dias_desde_ultima_luta,  # Descanso
        probabilidade_90_min,
        eh_penaltista,
        eh_em_casa,
        xG or xA if atacante,
    ]
    
    # Modelo treinado (XGBoost, LightGBM, etc)
    pontos_previstos = modelo.predict([features])[0]
    intervalo_confianca = [pontos_previstos * 0.7, pontos_previstos * 1.3]
    
    return pontos_previstos, intervalo_confianca
```

**Saída:**
- Pontos esperados: 12.5
- Intervalo (70% confiança): 8.8 - 16.3 pontos

---

## 8︠︡︠ VALIDAÇÃO DE ESCALAÇÃO (Checks Automáticos)

```python
def validar_escalacao(time_montado):
    erros = []
    avisos = []
    
    # ERROS CRÍTICOS (Impedem escala)
    if sum(1 for j in time_montado if j.time == "A" and j.posicao in ["GOL", "ZAG", "LAT"]) > 3:
        erros.append("Máximo 3 defensores do mesmo time")
    
    if sum(1 for j in time_montado if j.posicao == "ATA") < 2:
        erros.append("Mínimo 2 atacantes")
    
    if time_montado.capital < 0:
        erros.append("Saldo negativo!")
    
    # AVISOS (Não impedem, mas alertam)
    if sum(1 for j in time_montado if j.risco_cartao == "ALTO") > 1:
        avisos.append("2+ jogadores com risco alto de cartão")
    
    capitao = time_montado.capitao
    if not candidato_capitao_valido(capitao, rodada):
        avisos.append(f""{capitao.nome}" não atende checklist ótimo de capitão")
    
    return erros, avisos
```

---

## 9︠︡︠ MONÍTOR DE TENDÊNCIAS (Feedback Loop)

### Rastreamento Pós-Rodada

```python
def registrar_desempenho(rodada_numero, time_escalado, placar_final):
    for jogador in time_escalado:
        registro = {
            "rodada": rodada_numero,
            "jogador_id": jogador.id,
            "previsao_pts": jogador.pontos_previstos,
            "real_pts": jogador.pontos_reais,
            "erro_absoluto": abs(jogador.pontos_revistos - jogador.pontos_reais),
            "acertou_teto": jogador.pontos_reais > (jogador.pontos_previstos * 1.2),
            "nao_abriu": jogador.pontos_reais < (jogador.pontos_previstos * 0.5),
        }
        historico.append(registro)
    
    # Reentrenar modelo mensalmente com novos dados
    if len(historico) % 38 == 0:  # A cada rodada do campeonato
        modelo.retrain(historico)
```

**Benefícios:**
- Modelo fica mais preciso ao longo da temporada
- Identifica "clusters" de acerto (quem erra mais em qual tipo de jogo)

---

## 10 📄 DASHBOARD DE MONITORAMENTO

### Métricas por Rodada

| Métrica | Cálculo | Uso |
|---------|---------|-----|
| **MAE (Mean Absolute Error)** | Média de erro absoluto de predição | Avaliar qualidade do modelo |
| **Acurácia de Capitão** | % de rodadas em que capitão foi top-3 escalado | Validar heurística |
| **Taxa de Correlacao Violada** | % de rodadas com > 3 defensores do mesmo time | Evitar risco geométrico |
| **ROI de Valorização** | Ganho patrimonial / Rodadas apostadas | Validar estratégia |
| **Pontos vs Média da Liga** | Seu placar / Placar médio da liga | Benchmark competitivo |

---

## 🚀 ROADMAP DE IMPLEMENTAÇÃO

1. **Fase 1 (Mês 1-2):**
   - [ ] Coletor de dados scouts/xG
   - [ ] Cálculo de média por 90 minutos
   - [ ] Clusterização básica de rodadas

2. **Fase 2 (Mês 3-4):**
   - [ ] Modelo XGBoost de previsão
   - [ ] Reclassificação posicional (heat map)
   - [ ] Validator de escalacao

3. **Fase 3 (Mês 5+):**
   - [ ] Dashboard em tempo real
   - [ ] API para sugestão de capitão
   - [ ] Feedback loop e retrain mensal

---

**Esta é a metodologia de elite. Implemente gradualmente e meça impacto!**
