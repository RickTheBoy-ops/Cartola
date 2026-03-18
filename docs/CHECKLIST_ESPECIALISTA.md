# Guia Definitivo e Checklist do Especialista em Cartola FC

Além dos conceitos básicos, os especialistas "hardcore" adotam um mindset focado rigorosamente em dados, gestão de risco e antecipação por métricas avançadas. Construir um algoritmo de elite no Cartola (como nosso projeto) exige internalizar essas regras matemáticas e lógicas no código.

## 🧠 Mindset e Regras de Ouro
1. **Zero Clubismo:** A emoção é inimiga da pontuação. Escale jogadores do rival do seu time do coração se o confronto estatístico for favorável.
2. **Gestão de Patrimônio Inicial:** Nas 3 a 5 primeiras rodadas, a prioridade absoluta é **valorização**. Um cofre cheio (> C$ 150) permite escalar qualquer time no restante do campeonato sem restrições.
3. **Informação até o Último Minuto:** O mercado do Cartola é volátil. Reaja a lesões, viagens, desgaste físico e prováveis escalações até minutos antes do fechamento.

## 🚀 Nível Hardcore: Variáveis Avançadas e Modelagem
Para modelos de Machine Learning e Inteligência Artificial, as regras que diferenciam "chutes" de predições reais de elite são:

**1. Métricas Normalizadas por Tempo:**
- **Média por 90 minutos:** Usar média por jogo engana, pois jogadores que entram aos 40 minutos do 2º tempo derrubam sua média (sem ter tempo hábil para pontuar).
- **xG (Expected Goals) e xA (Expected Assists):** Cruzar quem finaliza muito com a qualidade da finalização. Acha o "azarado" que está prestes a explodir e mitar na próxima rodada.

**2. Matchups e Posição Tática Real:**
- **Scouts Cedidos por Posição / Mando:** O "Time X fora de casa costuma ceder 7 pts para laterais esquerdos". Usar esse cruzamento é exponencialmente superior a usar média bruta do jogador.
- **Posição Real vs Cartola:** Um "lateral" escalado frequentemente atua como "ponta" pelo mapa de calor. A IA deve reclassificá-lo de acordo com sua proatividade tática.
- **Árbitros:** Cruzar o perfil do juiz (punitivo vs complacente) com jogadores infratores (zagueiros) vs cavadores (pontas habilidosos).

**3. Clusterização da Rodada e Regras de Otimização:**
Times são categorizados matematicamente:
- **Grupo A (Amplo Favoritismo):** Garantir base da defesa para SG + 2 ofensivos que devem participar dos inevitáveis gols.
- **Grupo B (Jogos Abertos/Gols de ambos):** Evitar goleiros/zagueiros de ambos os times, focar apenas em ataque e meias.
- **Grupo C (Truncado/Para Fugir):** Focar no máximo em um zagueiro rebatedor. Zero atacantes escalados desse grupo.

**4. Gestão de Risco na Construção do Time:**
- **Limite de Correlação na Defesa:** Nunca mais de 3 defensores do mesmo time. "Morrer abraçado" com 5 jogadores da mesma defesa arruina a rodada por 1 gol cagado.
- **Lógica Estrita do Capitão:** O Capitão não é adivinhação. Ele deve cumprir o checklist algorítmico: *Amplo Favoritismo da equipe* + *Histórico de 90min em campo* + *Responsável pela bola parada (Pênalti/Falta/Escanteio)* + *Alta Média Recente*.
- **Reserva de Luxo Anti-Azar:** O reserva deve atuar em um jogo diferente do titular (para não ser corrompido pelo mesmo contexto lúgubre, como um jogo anulado por temporal) e, preferencialmente, atuar no domingo/segunda-feira para prover mais informação do que os jogos de sábado.

---

## ✅ Checklist de Escalação (Toda Rodada)

Siga esta rotina antes de confirmar seu time:

- [ ] **Análise de Clusters:** Identifiquei os Grupos A (SG quase certo), B (Chuva de gols) e C (Fuja).
- [ ] **Odds e IA Externa:** Cruzei as previsões das casas de aposta (Probs de Vitória, SG, Under/Over 2.5 gols) com os confrontos da rodada.
- [ ] **Checklist de Defesa:** Garanti que não passei do limite de 3 jogadores defensivos do mesmo time (evitando falência do SG).
- [ ] **Matchup por Posição:** Verifiquei se meus Laterais e Atacantes vão jogar contra defesas que estatisticamente cedem *muitos pontos* nas suas respectivas posições.
- [ ] **Checklist do Capitão:** Meu capitão bate pênalti/falta, atua num time Favorito (Grupo A) e joga 90 minutos?
- [ ] **Reserva Estratégica:** Meu reserva de luxo pertence a um confronto diferente do titular para evitar "correlação de azares"?
- [ ] **Minutos Finais:** Conferi desfalques, time misto de última hora e condições climáticas absurdas a 30 mins do fechamento?
