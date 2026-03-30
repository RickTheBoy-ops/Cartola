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
        return (
            self.goleiros
            + self.zagueiros
            + self.laterais
            + self.meias
            + self.atacantes
            + self.tecnicos
        )


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
    6: 'tecnico',
}

# Máximo padrão de jogadores do mesmo clube (pode ser sobrescrito no __init__)
MAX_MESMO_CLUBE_PADRAO = 3


class GeneticTeamOptimizer:
    """Otimizador de escalação usando Algoritmo Genético.

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
        penalidade_variancia: bool = True,
        partidas_df: pd.DataFrame = None,
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

        # ── Regra Anti-Confronto (regra #1 dos cartoleiros) ──────────
        # Monta um set de pares {frozenset({clube_a, clube_b})} que se enfrentam
        self.confrontos_set: set = set()
        if partidas_df is not None and len(partidas_df) > 0:
            for _, row in partidas_df.iterrows():
                c_a = row.get('clube_casa_id') or row.get('clube_id_a')
                c_b = row.get('clube_visitante_id') or row.get('clube_id_b')
                if c_a and c_b and c_a != c_b:
                    self.confrontos_set.add(frozenset({int(c_a), int(c_b)}))
        logger.info(
            "🚫 Anti-confronto: %d confrontos mapeados", len(self.confrontos_set)
        )

        # Filtrar atletas válidos
        atletas_validos = filtrar_atletas_validos(atletas_df)

        # Merge atletas com predições
        cols_to_merge = ['atleta_id', 'predicao']
        if 'fitness_score' in predicoes.columns:
            cols_to_merge.append('fitness_score')
        if 'mpv' in predicoes.columns:
            cols_to_merge.append('mpv')

        cols_to_drop = [
            c
            for c in cols_to_merge
            if c in atletas_validos.columns and c != 'atleta_id'
        ]
        if cols_to_drop:
            atletas_validos = atletas_validos.drop(columns=cols_to_drop)

        self.data = atletas_validos.merge(
            predicoes[cols_to_merge],
            on='atleta_id',
            how='inner',
        )
        self.fitness_col = (
            'fitness_score' if 'fitness_score' in self.data.columns else 'predicao'
        )

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
            "🧬 Otimizador inicializado - Formação: %s, Patrimônio: C$ %.2f, Atletas válidos: %d",
            formacao,
            patrimonio,
            len(self.data),
        )

    def _verificar_atletas_suficientes(self):
        """Verifica se há atletas suficientes para cada posição da formação."""
        necessidades = {
            1: self.formacao.goleiros,
            2: self.formacao.laterais,
            3: self.formacao.zagueiros,
            4: self.formacao.meias,
            5: self.formacao.atacantes,
            6: self.formacao.tecnicos,
        }

        for pos_id, qtd_necessaria in necessidades.items():
            disponivel = len(self.atletas_por_posicao.get(pos_id, []))
            if disponivel < qtd_necessaria:
                logger.warning(
                    "⚠️ Posição %s: necessários %d, disponíveis %d",
                    POSICOES[pos_id],
                    qtd_necessaria,
                    disponivel,
                )

    def create_random_team(self) -> List[Dict]:
        """Cria time aleatório válido sem duplicatas."""
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
            disponiveis = [
                a
                for a in self.atletas_por_posicao[pos_id]
                if a['atleta_id'] not in used_ids
            ]
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
        """Gera chave única para um time (para cache de fitness)."""
        return tuple(sorted(a['atleta_id'] for a in team))

    def fitness(self, team: List[Dict]) -> float:
        """Calcula o fitness de um time.

        Regras principais:
        - Time deve ter exatamente o número de jogadores da formação
        - Não pode ultrapassar o patrimônio (hard constraint)
        - Penaliza concentração por clube, alta variância, duplicatas, confrontos, etc.
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

        # Hard constraint: time precisa ter exatamente o número de jogadores da formação
        expected_players = self.formacao.total_jogadores()
        if len(team) != expected_players:
            score = 0.0
            self._fitness_cache[key] = score
            return score

        # === Bônus do Capitão: MEI (4) ou ATA (5) com maior score tem pontos dobrados ===
        capitao_bonus = max(
            (
                a.get(self.fitness_col, 0)
                for a in team
                if a.get('posicao_id') in [4, 5]
            ),
            default=0.0,
        )
        total_pontos += capitao_bonus

        # === Hard constraint: ultrapassar patrimônio zera o fitness ===
        if total_preco > self.patrimonio:
            score = 0.0
            self._fitness_cache[key] = score
            return score

        # === Bônus: uso eficiente do patrimônio ===
        uso_patrimonio = total_preco / self.patrimonio if self.patrimonio > 0 else 0
        if uso_patrimonio >= 0.9:
            bonus = 2.0
        elif uso_patrimonio >= 0.8:
            bonus = 1.0
        else:
            bonus = 0.0

        # === Penalidade: muitos jogadores do mesmo clube e risco de SGs ===
        clubes: Dict[int, int] = {}
        clubes_defesa: Dict[int, int] = {}
        for a in team:
            clube_id = a.get('clube_id', 0)
            pos = a.get('posicao_id', 0)

            clubes[clube_id] = clubes.get(clube_id, 0) + 1
            if pos in [1, 2, 3]:  # Defesa
                clubes_defesa[clube_id] = clubes_defesa.get(clube_id, 0) + 1

        penalidade_clube = 0.0
        for clube_id, count in clubes.items():
            if count > self.max_mesmo_clube:
                # Penalidade forte para inviabilizar times com muitos jogadores do mesmo clube
                penalidade_clube += (count - self.max_mesmo_clube) * 50.0

        # Regra defensiva: respeita max_mesmo_clube, com teto=2 para preservar SG coletiva
        max_def = min(2, self.max_mesmo_clube)
        for clube_id, count_def in clubes_defesa.items():
            if count_def > max_def:
                penalidade_clube += (count_def - max_def) * 12.0

        # === Penalidade: alta variância (atletas instáveis) ===
        penalidade_variancia = 0.0
        if self.penalidade_variancia:
            for a in team:
                std_pred = a.get('predicao_std', 0)
                if std_pred > 4.0:
                    penalidade_variancia += (std_pred - 4.0) * 0.5

        # === Penalidade: jogadores duplicados ===
        ids = [a.get('atleta_id') for a in team]
        duplicatas = len(ids) - len(set(ids))
        penalidade_duplicata = duplicatas * 50.0

        # === Penalidade Anti-Confronto (regra #1 dos cartoleiros) ===
        penalidade_confronto = 0.0
        if self.confrontos_set:
            clubes_no_time = [
                (a.get('clube_id', 0), a.get('posicao_id', 0)) for a in team
            ]
            for i in range(len(clubes_no_time)):
                for j in range(i + 1, len(clubes_no_time)):
                    c_i, pos_i = clubes_no_time[i]
                    c_j, pos_j = clubes_no_time[j]
                    if c_i != c_j and frozenset({int(c_i), int(c_j)}) in self.confrontos_set:
                        penalidade_confronto += 30.0
                        is_atk_i = pos_i in [4, 5]
                        is_def_i = pos_i in [1, 2, 3]
                        is_atk_j = pos_j in [4, 5]
                        is_def_j = pos_j in [1, 2, 3]
                        if (is_atk_i and is_def_j) or (is_atk_j and is_def_i):
                            penalidade_confronto += 20.0

        # === Penalidade: Não-Titulares ===
        penalidade_status = 0.0
        for a in team:
            if int(a.get('status_id', 7)) != 7:
                penalidade_status += 100.0

        # === Penalidade: Custo-Benefício Ruim & Regra de Ouro Miteira ===
        penalidade_custo_beneficio = 0.0
        penalidade_ausencia_elite = 0.0
        for a in team:
            p = a.get('preco_num', a.get('preco', 0))
            m = a.get('media_num', a.get('media', 0))
            pos = a.get('posicao_id', 0)
            pred = a.get('predicao', a.get('predicao_ajustada', 0))

            if p > 5.0 and m < 4.0:
                penalidade_custo_beneficio += 15.0

            if pos == 6 and pred < 6.5:
                penalidade_ausencia_elite += 20.0

            if pos == 5 and pred < 7.0:
                penalidade_ausencia_elite += 15.0

        score = (
            total_pontos
            + bonus
            - penalidade_clube
            - penalidade_variancia
            - penalidade_duplicata
            - penalidade_confronto
            - penalidade_status
            - penalidade_custo_beneficio
            - penalidade_ausencia_elite
        )
        self._fitness_cache[key] = score
        return score

    def crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """Crossover por posição com verificação de duplicatas."""
        child: List[Dict] = []
        used_ids = set()

        positions_count = {
            1: self.formacao.goleiros,
            2: self.formacao.laterais,
            3: self.formacao.zagueiros,
            4: self.formacao.meias,
            5: self.formacao.atacantes,
            6: self.formacao.tecnicos,
        }

        for pos_id, count in positions_count.items():
            if count == 0:
                continue

            parent1_pos = [a for a in parent1 if a['posicao_id'] == pos_id]
            parent2_pos = [a for a in parent2 if a['posicao_id'] == pos_id]

            for i in range(count):
                if random.random() < 0.5 and i < len(parent1_pos):
                    candidate = parent1_pos[i]
                elif i < len(parent2_pos):
                    candidate = parent2_pos[i]
                elif i < len(parent1_pos):
                    candidate = parent1_pos[i]
                else:
                    disponivel = [
                        a
                        for a in self.atletas_por_posicao.get(pos_id, [])
                        if a['atleta_id'] not in used_ids
                    ]
                    candidate = random.choice(disponivel) if disponivel else None

                if candidate and candidate['atleta_id'] not in used_ids:
                    child.append(candidate)
                    used_ids.add(candidate['atleta_id'])
                else:
                    disponivel = [
                        a
                        for a in self.atletas_por_posicao.get(pos_id, [])
                        if a['atleta_id'] not in used_ids
                    ]
                    if disponivel:
                        novo = random.choice(disponivel)
                        child.append(novo)
                        used_ids.add(novo['atleta_id'])

        return child

    def mutate(self, team: List[Dict]) -> List[Dict]:
        """Mutação: substitui jogador aleatório por outro da mesma posição."""
        if random.random() > self.mutation_rate:
            return team

        team_copy = team.copy()
        used_ids = {a['atleta_id'] for a in team_copy}

        mutable_positions = [
            p for p in [1, 2, 3, 4, 5, 6] if len(self.atletas_por_posicao.get(p, [])) > 1
        ]
        if not mutable_positions:
            return team_copy
        pos_to_mutate = random.choice(mutable_positions)

        team_players = [a for a in team_copy if a['posicao_id'] == pos_to_mutate]

        if team_players:
            player_to_replace = random.choice(team_players)

            available = [
                a
                for a in self.atletas_por_posicao.get(pos_to_mutate, [])
                if a['atleta_id'] not in used_ids
            ]

            if available:
                new_player = random.choice(available)
                idx = team_copy.index(player_to_replace)
                team_copy[idx] = new_player

        return team_copy

    def optimize(self) -> Tuple[List[Dict], Dict]:
        """Executa algoritmo genético com elitismo e torneio."""
        population = [self.create_random_team() for _ in range(self.population_size)]

        best_team: List[Dict] | None = None
        best_fitness = -float('inf')

        logger.info(
            "🧬 Iniciando otimização - %d gerações, pop=%d",
            self.generations,
            self.population_size,
        )

        for generation in range(self.generations):
            fitness_scores = [(team, self.fitness(team)) for team in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            current_best = fitness_scores[0]

            if current_best[1] > best_fitness:
                best_team = current_best[0]
                best_fitness = current_best[1]

            if generation % 20 == 0:
                total_price = sum(a.get('preco', 0) for a in current_best[0])
                total_points = sum(a.get('predicao', 0) for a in current_best[0])
                logger.info(
                    "  Gen %3d: Fitness=%.2f, Pontos=%.2f, Preço=C$%.2f",
                    generation,
                    current_best[1],
                    total_points,
                    total_price,
                )

            next_generation = [team for team, _ in fitness_scores[: self.elite_size]]

            while len(next_generation) < self.population_size:
                tournament_size = 5
                tournament = random.sample(
                    fitness_scores, min(tournament_size, len(fitness_scores))
                )
                parent1 = max(tournament, key=lambda x: x[1])[0]

                tournament = random.sample(
                    fitness_scores, min(tournament_size, len(fitness_scores))
                )
                parent2 = max(tournament, key=lambda x: x[1])[0]

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                next_generation.append(child)

            population = next_generation

        assert best_team is not None, "Algoritmo genético não encontrou equipe válida"

        total_price = sum(a.get('preco', 0) for a in best_team)
        total_points = sum(a.get('predicao', 0) for a in best_team)

        stats = {
            'total_pontos_preditos': total_points,
            'total_preco': total_price,
            'patrimonio_usado': (total_price / self.patrimonio) * 100
            if self.patrimonio > 0
            else 0,
            'fitness': best_fitness,
            'formacao': self.formacao_nome,
            'cache_size': len(self._fitness_cache),
        }

        validacao = validar_time(best_team, self.patrimonio, self.formacao_nome)
        if not validacao['valido']:
            logger.warning("⚠️ Time gerado com problemas: %s", validacao['erros'])

        logger.info("\n%s", "=" * 50)
        logger.info("🏆 OTIMIZAÇÃO CONCLUÍDA")
        logger.info("Pontos Preditos: %.2f", total_points)
        logger.info("Preço Total: C$ %.2f", total_price)
        logger.info("Patrimônio Usado: %.1f%%", stats['patrimonio_usado'])
        logger.info("Cache de fitness: %d avaliações", stats['cache_size'])
        logger.info("%s\n", "=" * 50)

        return best_team, stats

    def format_team_output(self, team: List[Dict]) -> pd.DataFrame:
        """Formata time otimizado para exibição com marcação do Capitão (C)."""

        melhor_score = -1.0
        capitao_idx = -1
        score_col = getattr(self, 'fitness_col', 'predicao')

        for i, a in enumerate(team):
            if a.get('posicao_id') in [4, 5]:
                score_estimado = a.get(score_col, 0)
                if a.get('mando_casa', 0) == 1:
                    score_estimado *= 1.25

                if score_estimado > melhor_score:
                    melhor_score = score_estimado
                    capitao_idx = i

        if capitao_idx == -1:
            linha = [i for i, a in enumerate(team) if a.get('posicao_id') != 6]
            if linha:
                capitao_idx = max(linha, key=lambda i: team[i].get(score_col, 0))

        team_display = [dict(a) for a in team]
        if capitao_idx != -1:
            team_display[capitao_idx]['apelido'] = (
                str(team_display[capitao_idx].get('apelido', '')) + ' (C)'
            )

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
            'mpv': 'MPV',
        }

        if getattr(self, 'fitness_col', 'predicao') != 'predicao':
            col_map[self.fitness_col] = 'Score Otimização'

        ordem_exibicao = [
            'apelido',
            'posicao_nome',
            'clube_id',
            'preco',
            'predicao',
            getattr(self, 'fitness_col', 'predicao'),
            'mpv',
            'media',
        ]

        cols_saida = [c for c in ordem_exibicao if c in df.columns and c in col_map]

        df = df[cols_saida]
        df.columns = [col_map[c] for c in cols_saida]

        return df
