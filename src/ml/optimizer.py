import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
import logging

from src.utils.validators import validar_time, filtrar_atletas_validos

logger = logging.getLogger(__name__)


@dataclass
class Formacao:
    """Define uma formação tática"""
    goleiros: int = 1
    zagueiros: int = 2
    laterais: int = 2
    meias: int = 3
    atacantes: int = 3
    tecnicos: int = 1

    def total_jogadores(self) -> int:
        return (self.goleiros + self.zagueiros + self.laterais +
                self.meias + self.atacantes + self.tecnicos)


# Formações disponíveis
FORMACOES = {
    '3-4-3': Formacao(1, 3, 0, 4, 3, 1),
    '3-5-2': Formacao(1, 3, 0, 5, 2, 1),
    '4-3-3': Formacao(1, 2, 2, 3, 3, 1),
    '4-4-2': Formacao(1, 2, 2, 4, 2, 1),
    '4-5-1': Formacao(1, 2, 2, 5, 1, 1),
    '5-3-2': Formacao(1, 3, 2, 3, 2, 1),
    '5-4-1': Formacao(1, 3, 2, 4, 1, 1),
}

# Mapeamento posicao_id -> nome
POSICOES = {
    1: 'goleiro',
    2: 'lateral',
    3: 'zagueiro',
    4: 'meia',
    5: 'atacante',
    6: 'tecnico'
}

# Máximo padrão de jogadores do mesmo clube (pode ser sobrescrito no __init__)
MAX_MESMO_CLUBE_PADRAO = 3


class GeneticTeamOptimizer:
    """
    Otimizador de escalação usando Algoritmo Genético.
    Melhorias:
    - Filtra atletas inválidos (lesionados, suspensos)
    - Penaliza concentração de jogadores do mesmo clube
    - Verifica duplicatas no time
    - Cache de fitness para times idênticos
    """

    def __init__(
        self,
        atletas_df: pd.DataFrame,
        predicoes: pd.DataFrame,
        patrimonio: float,
        formacao: str = '4-3-3',
        population_size: int = 250,
        generations: int = 150,
        mutation_rate: float = 0.20,
        elite_size: int = 20,
        max_mesmo_clube: int = 3,
        penalidade_variancia: bool = True
    ):
        self.patrimonio = patrimonio
        self.formacao = FORMACOES[formacao]
        self.formacao_nome = formacao
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_mesmo_clube = max_mesmo_clube
        self.penalidade_variancia = penalidade_variancia

        # Filtrar atletas válidos
        atletas_validos = filtrar_atletas_validos(atletas_df)

        # Merge atletas com predições
        cols_to_merge = ['atleta_id', 'predicao']
        if 'fitness_score' in predicoes.columns:
            cols_to_merge.append('fitness_score')
        if 'mpv' in predicoes.columns:
            cols_to_merge.append('mpv')

        self.data = atletas_validos.merge(
            predicoes[cols_to_merge],
            on='atleta_id',
            how='inner'
        )
        self.fitness_col = 'fitness_score' if 'fitness_score' in self.data.columns else 'predicao'

        # Garantir que 'media' existe
        if 'media' not in self.data.columns:
            self.data['media'] = self.data.get('predicao', 0)

        # Separar por posição
        self.atletas_por_posicao = {}
        for pos_id in POSICOES.keys():
            pos_atletas = self.data[self.data['posicao_id'] == pos_id].to_dict('records')
            self.atletas_por_posicao[pos_id] = pos_atletas

        # Verificar se temos atletas suficientes por posição
        self._verificar_atletas_suficientes()

        # Cache de fitness
        self._fitness_cache = {}

        logger.info(
            f"🧬 Otimizador inicializado - Formação: {formacao}, "
            f"Patrimônio: C$ {patrimonio:.2f}, "
            f"Atletas válidos: {len(self.data)}"
        )

    def _verificar_atletas_suficientes(self):
        """Verifica se há atletas suficientes para cada posição da formação"""
        necessidades = {
            1: self.formacao.goleiros,
            2: self.formacao.laterais,
            3: self.formacao.zagueiros,
            4: self.formacao.meias,
            5: self.formacao.atacantes,
            6: self.formacao.tecnicos
        }

        for pos_id, qtd_necessaria in necessidades.items():
            disponivel = len(self.atletas_por_posicao.get(pos_id, []))
            if disponivel < qtd_necessaria:
                logger.warning(
                    f"⚠️ Posição {POSICOES[pos_id]}: necessários {qtd_necessaria}, "
                    f"disponíveis {disponivel}"
                )

    def create_random_team(self) -> List[Dict]:
        """Cria time aleatório válido sem duplicatas"""
        team = []
        used_ids = set()

        posicoes_ordem = [
            (1, self.formacao.goleiros),
            (3, self.formacao.zagueiros),
            (2, self.formacao.laterais),
            (4, self.formacao.meias),
            (5, self.formacao.atacantes),
            (6, self.formacao.tecnicos),
        ]

        for pos_id, count in posicoes_ordem:
            if count == 0:
                continue
            disponiveis = [a for a in self.atletas_por_posicao[pos_id]
                           if a['atleta_id'] not in used_ids]
            if len(disponiveis) < count:
                # Fallback: usar todos disponíveis
                selecionados = disponiveis
            else:
                selecionados = random.sample(disponiveis, count)
            for a in selecionados:
                used_ids.add(a['atleta_id'])
            team.extend(selecionados)

        return team

    def _team_key(self, team: List[Dict]) -> tuple:
        """Gera chave única para um time (para cache de fitness)"""
        return tuple(sorted(a['atleta_id'] for a in team))

    def fitness(self, team: List[Dict]) -> float:
        """
        Função de fitness com penalizações:
        - Ultrapassar patrimônio
        - Concentração de jogadores do mesmo clube
        - Jogadores duplicados
        - Bônus do capitão (MEI/ATA com maior score tem pontos dobrados)
        """
        # Cache
        key = self._team_key(team)
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        # Limpar cache se crescer demais (evitar vazamento de memória)
        if len(self._fitness_cache) > 50_000:
            self._fitness_cache.clear()

        total_pontos = sum(atleta.get(self.fitness_col, 0) for atleta in team)
        total_preco = sum(atleta.get('preco', 0) for atleta in team)

        # === Bônus do Capitão: MEI (4) ou ATA (5) com maior score tem pontos dobrados ===
        # Inclui aqui para que o AG otimize o time COM o capitão, não SÓ na exibição
        capitao_bonus = max(
            (a.get(self.fitness_col, 0) for a in team if a.get('posicao_id') in [4, 5]),
            default=0.0
        )
        total_pontos += capitao_bonus  # +1x o score do melhor MEI/ATA = pontos dobrados

        # === Penalidade: ultrapassar patrimônio ===
        if total_preco > self.patrimonio:
            penalidade = (total_preco - self.patrimonio) * 10
            score = max(0, total_pontos - penalidade)
            self._fitness_cache[key] = score
            return score

        # === Bônus: uso eficiente do patrimônio ===
        uso_patrimonio = total_preco / self.patrimonio if self.patrimonio > 0 else 0
        if uso_patrimonio >= 0.9:
            bonus = 2.0
        elif uso_patrimonio >= 0.8:
            bonus = 1.0
        else:
            bonus = 0

        # === Penalidade: muitos jogadores do mesmo clube e risco de SGs ===
        clubes = {}
        clubes_defesa = {}
        for a in team:
            clube_id = a.get('clube_id', 0)
            pos = a.get('posicao_id', 0)

            clubes[clube_id] = clubes.get(clube_id, 0) + 1
            if pos in [1, 2, 3]:  # Defesa
                clubes_defesa[clube_id] = clubes_defesa.get(clube_id, 0) + 1

        penalidade_clube = 0
        for clube_id, count in clubes.items():
            if count > self.max_mesmo_clube:
                penalidade_clube += (count - self.max_mesmo_clube) * 5

        # Regra defensiva: respeita max_mesmo_clube, com teto=2 para preservar SG coletiva
        max_def = min(2, self.max_mesmo_clube)
        for clube_id, count_def in clubes_defesa.items():
            if count_def > max_def:
                penalidade_clube += (count_def - max_def) * 12

        # === Penalidade: alta variância (atletas instáveis) ===
        penalidade_variancia = 0
        if self.penalidade_variancia:
            for a in team:
                std_pred = a.get('predicao_std', 0)
                if std_pred > 4.0:
                    penalidade_variancia += (std_pred - 4.0) * 0.5

        # === Penalidade: jogadores duplicados ===
        ids = [a.get('atleta_id') for a in team]
        duplicatas = len(ids) - len(set(ids))
        penalidade_duplicata = duplicatas * 50

        score = total_pontos + bonus - penalidade_clube - penalidade_variancia - penalidade_duplicata
        self._fitness_cache[key] = score
        return score

    def crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """Crossover por posição com verificação de duplicatas"""
        child = []
        used_ids = set()

        positions_count = {
            1: self.formacao.goleiros,
            2: self.formacao.laterais,
            3: self.formacao.zagueiros,
            4: self.formacao.meias,
            5: self.formacao.atacantes,
            6: self.formacao.tecnicos
        }

        for pos_id, count in positions_count.items():
            if count == 0:
                continue

            parent1_pos = [a for a in parent1 if a['posicao_id'] == pos_id]
            parent2_pos = [a for a in parent2 if a['posicao_id'] == pos_id]

            for i in range(count):
                # Tentar herdar de um dos pais sem duplicata
                if random.random() < 0.5 and i < len(parent1_pos):
                    candidate = parent1_pos[i]
                elif i < len(parent2_pos):
                    candidate = parent2_pos[i]
                elif i < len(parent1_pos):
                    candidate = parent1_pos[i]
                else:
                    # Fallback: pegar aleatório da posição
                    disponivel = [a for a in self.atletas_por_posicao.get(pos_id, [])
                                  if a['atleta_id'] not in used_ids]
                    candidate = random.choice(disponivel) if disponivel else None

                if candidate and candidate['atleta_id'] not in used_ids:
                    child.append(candidate)
                    used_ids.add(candidate['atleta_id'])
                else:
                    # Substituir por outro da mesma posição
                    disponivel = [a for a in self.atletas_por_posicao.get(pos_id, [])
                                  if a['atleta_id'] not in used_ids]
                    if disponivel:
                        novo = random.choice(disponivel)
                        child.append(novo)
                        used_ids.add(novo['atleta_id'])

        return child

    def mutate(self, team: List[Dict]) -> List[Dict]:
        """Mutação: substitui jogador aleatório por outro da mesma posição.

        Inclui todas as posições (GOL, LAT, ZAG, MEI, ATA, TEC) para não
        fixar goleiro/técnico subótimos nas primeiras gerações.
        """
        if random.random() > self.mutation_rate:
            return team

        team_copy = team.copy()
        used_ids = {a['atleta_id'] for a in team_copy}

        # Todas as posições são mutáveis
        mutable_positions = [p for p in [1, 2, 3, 4, 5, 6]
                             if len(self.atletas_por_posicao.get(p, [])) > 1]
        if not mutable_positions:
            return team_copy
        pos_to_mutate = random.choice(mutable_positions)

        team_players = [a for a in team_copy if a['posicao_id'] == pos_to_mutate]

        if team_players:
            player_to_replace = random.choice(team_players)

            available = [a for a in self.atletas_por_posicao.get(pos_to_mutate, [])
                         if a['atleta_id'] not in used_ids]

            if available:
                new_player = random.choice(available)
                idx = team_copy.index(player_to_replace)
                team_copy[idx] = new_player

        return team_copy

    def optimize(self) -> Tuple[List[Dict], Dict]:
        """Executa algoritmo genético com elitismo e torneio"""
        population = [self.create_random_team() for _ in range(self.population_size)]

        best_team = None
        best_fitness = -float('inf')

        logger.info(f"🧬 Iniciando otimização - {self.generations} gerações, pop={self.population_size}")

        for generation in range(self.generations):
            # Calcular fitness
            fitness_scores = [(team, self.fitness(team)) for team in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            current_best = fitness_scores[0]

            if current_best[1] > best_fitness:
                best_team = current_best[0]
                best_fitness = current_best[1]

            # Log a cada 20 gerações
            if generation % 20 == 0:
                total_price = sum(a.get('preco', 0) for a in current_best[0])
                total_points = sum(a.get('predicao', 0) for a in current_best[0])
                logger.info(
                    f"  Gen {generation:3d}: "
                    f"Fitness={current_best[1]:.2f}, "
                    f"Pontos={total_points:.2f}, "
                    f"Preço=C${total_price:.2f}"
                )

            # Seleção (elitismo)
            next_generation = [team for team, _ in fitness_scores[:self.elite_size]]

            # Crossover e mutação
            while len(next_generation) < self.population_size:
                tournament_size = 5
                tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
                parent1 = max(tournament, key=lambda x: x[1])[0]

                tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
                parent2 = max(tournament, key=lambda x: x[1])[0]

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                next_generation.append(child)

            population = next_generation

        # Estatísticas finais
        total_price = sum(a.get('preco', 0) for a in best_team)
        total_points = sum(a.get('predicao', 0) for a in best_team)

        stats = {
            'total_pontos_preditos': total_points,
            'total_preco': total_price,
            'patrimonio_usado': (total_price / self.patrimonio) * 100 if self.patrimonio > 0 else 0,
            'fitness': best_fitness,
            'formacao': self.formacao_nome,
            'cache_size': len(self._fitness_cache)
        }

        # Validar time final
        validacao = validar_time(best_team, self.patrimonio, self.formacao_nome)
        if not validacao['valido']:
            logger.warning(f"⚠️ Time gerado com problemas: {validacao['erros']}")

        logger.info(f"\n{'=' * 50}")
        logger.info(f"🏆 OTIMIZAÇÃO CONCLUÍDA")
        logger.info(f"Pontos Preditos: {total_points:.2f}")
        logger.info(f"Preço Total: C$ {total_price:.2f}")
        logger.info(f"Patrimônio Usado: {stats['patrimonio_usado']:.1f}%")
        logger.info(f"Cache de fitness: {stats['cache_size']} avaliações")
        logger.info(f"{'=' * 50}\n")

        return best_team, stats

    def format_team_output(self, team: List[Dict]) -> pd.DataFrame:
        """Formata time otimizado para exibição com marcação do Capitão (C)"""
        
        # === Hardcore Rule: Gestão do Capitão ===
        melhor_score = -1
        capitao_idx = -1
        score_col = getattr(self, 'fitness_col', 'predicao')
        
        for i, a in enumerate(team):
            if a.get('posicao_id') in [4, 5]: # Foco em Meia (4) ou Atacante (5)
                # Bônus se jogar em casa, essencial para capitão
                score_estimado = a.get(score_col, 0)
                if a.get('mando_casa', 0) == 1:
                    score_estimado *= 1.25 
                
                if score_estimado > melhor_score:
                    melhor_score = score_estimado
                    capitao_idx = i
                    
        # Fallback: Se não achar, pega o maior score da linha (exceto tecnico)
        if capitao_idx == -1:
            linha = [i for i, a in enumerate(team) if a.get('posicao_id') != 6]
            if linha:
                capitao_idx = max(linha, key=lambda i: team[i].get(score_col, 0))

        team_display = [dict(a) for a in team]
        if capitao_idx != -1:
            team_display[capitao_idx]['apelido'] = str(team_display[capitao_idx].get('apelido', '')) + ' (C)'
            # Pontos do capitão já são dobrados no fitness().
            # Aqui só sinalizamos na exibição; não multiplicamos 'predicao' de novo.

        df = pd.DataFrame(team_display)

        df['posicao_nome'] = df['posicao_id'].map(POSICOES)

        df = df.sort_values(['posicao_id', 'predicao'], ascending=[True, False])

        col_map = {
            'apelido': 'Atleta',
            'posicao_nome': 'Posição',
            'clube_id': 'Clube',
            'preco': 'Preço (C$)',
            'predicao': 'Pred (Pts)',
            'media': 'Média',
            'mpv': 'MPV'
        }
        
        if getattr(self, 'fitness_col', 'predicao') != 'predicao':
            col_map[self.fitness_col] = 'Score Otimização'

        ordem_exibicao = ['apelido', 'posicao_nome', 'clube_id', 'preco', 'predicao', getattr(self, 'fitness_col', 'predicao'), 'mpv', 'media']
        
        cols_saida = [c for c in ordem_exibicao if c in df.columns and c in col_map]
        
        df = df[cols_saida]
        df.columns = [col_map[c] for c in cols_saida]

        return df
