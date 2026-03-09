import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
import logging

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

class GeneticTeamOptimizer:
    """
    Otimizador de escalação usando Algoritmo Genético
    """
    
    def __init__(
        self,
        atletas_df: pd.DataFrame,
        predicoes: pd.DataFrame,
        patrimonio: float,
        formacao: str = '4-3-3',
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.15,
        elite_size: int = 10
    ):
        self.atletas_df = atletas_df.copy()
        self.predicoes = predicoes.copy()
        self.patrimonio = patrimonio
        self.formacao = FORMACOES[formacao]
        self.formacao_nome = formacao
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Merge atletas com predições
        self.data = self.atletas_df.merge(
            self.predicoes[['atleta_id', 'predicao']],
            on='atleta_id',
            how='inner'
        )
        
        # Separar por posição
        self.atletas_por_posicao = {
            pos_id: self.data[self.data['posicao_id'] == pos_id].to_dict('records')
            for pos_id in POSICOES.keys()
        }
        
        logger.info(f"Otimizador inicializado - Formação: {formacao}, Patrimônio: C$ {patrimonio:.2f}")
    
    def create_random_team(self) -> List[Dict]:
        """Cria time aleatório válido"""
        team = []
        
        # Goleiro
        team.extend(random.sample(self.atletas_por_posicao[1], self.formacao.goleiros))
        
        # Zagueiros
        team.extend(random.sample(self.atletas_por_posicao[3], self.formacao.zagueiros))
        
        # Laterais
        if self.formacao.laterais > 0:
            team.extend(random.sample(self.atletas_por_posicao[2], self.formacao.laterais))
        
        # Meias
        team.extend(random.sample(self.atletas_por_posicao[4], self.formacao.meias))
        
        # Atacantes
        team.extend(random.sample(self.atletas_por_posicao[5], self.formacao.atacantes))
        
        # Técnico
        team.extend(random.sample(self.atletas_por_posicao[6], self.formacao.tecnicos))
        
        return team
    
    def fitness(self, team: List[Dict]) -> float:
        """
        Função de fitness (objetivo a maximizar)
        
        Considera:
        - Pontuação predita total
        - Penalidade se ultrapassar patrimônio
        - Bônus para times dentro do orçamento
        """
        total_pontos = sum(atleta['predicao'] for atleta in team)
        total_preco = sum(atleta['preco'] for atleta in team)
        
        # Penalidade severa se ultrapassar patrimônio
        if total_preco > self.patrimonio:
            penalidade = (total_preco - self.patrimonio) * 10  # Penalidade pesada
            return max(0, total_pontos - penalidade)
        
        # Bônus por usar o patrimônio eficientemente (90-100% do disponível)
        uso_patrimonio = total_preco / self.patrimonio
        if uso_patrimonio >= 0.9:
            bonus = 2.0
        elif uso_patrimonio >= 0.8:
            bonus = 1.0
        else:
            bonus = 0
        
        return total_pontos + bonus
    
    def crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """
        Crossover entre dois times (pais) por posição
        """
        child = []
        
        # Para cada posição, escolhe aleatoriamente do pai1 ou pai2
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
            
            # Jogadores dessa posição em cada pai
            parent1_pos = [a for a in parent1 if a['posicao_id'] == pos_id]
            parent2_pos = [a for a in parent2 if a['posicao_id'] == pos_id]
            
            # Escolher aleatoriamente de qual pai herdar
            for i in range(count):
                if random.random() < 0.5:
                    child.append(parent1_pos[i])
                else:
                    child.append(parent2_pos[i])
        
        return child
    
    def mutate(self, team: List[Dict]) -> List[Dict]:
        """
        Mutação: substitui jogadores aleatórios por outros da mesma posição
        """
        if random.random() > self.mutation_rate:
            return team
        
        team_copy = team.copy()
        
        # Escolher posição aleatória para mutar (exceto goleiro e técnico para maior estabilidade)
        mutable_positions = [2, 3, 4, 5]
        pos_to_mutate = random.choice(mutable_positions)
        
        # Encontrar jogadores dessa posição no time
        team_players = [a for a in team_copy if a['posicao_id'] == pos_to_mutate]
        
        if team_players:
            # Escolher um jogador para substituir
            player_to_replace = random.choice(team_players)
            
            # Escolher substituto da mesma posição
            available = [a for a in self.atletas_por_posicao[pos_to_mutate] 
                        if a['atleta_id'] != player_to_replace['atleta_id']]
            
            if available:
                new_player = random.choice(available)
                
                # Substituir
                idx = team_copy.index(player_to_replace)
                team_copy[idx] = new_player
        
        return team_copy
    
    def optimize(self) -> Tuple[List[Dict], Dict]:
        """
        Executa algoritmo genético
        """
        # População inicial
        population = [self.create_random_team() for _ in range(self.population_size)]
        
        best_team = None
        best_fitness = 0
        
        logger.info(f"Iniciando otimização - {self.generations} gerações")
        
        for generation in range(self.generations):
            # Calcular fitness de toda população
            fitness_scores = [(team, self.fitness(team)) for team in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Melhor desta geração
            current_best = fitness_scores[0]
            
            if current_best[1] > best_fitness:
                best_team = current_best[0]
                best_fitness = current_best[1]
            
            # Log progresso
            if generation % 10 == 0:
                total_price = sum(a['preco'] for a in current_best[0])
                total_points = sum(a['predicao'] for a in current_best[0])
                logger.info(
                    f"Geração {generation}: "
                    f"Fitness={current_best[1]:.2f}, "
                    f"Pontos={total_points:.2f}, "
                    f"Preço=C${total_price:.2f}"
                )
            
            # Seleção (elitismo + torneio)
            next_generation = [team for team, _ in fitness_scores[:self.elite_size]]
            
            # Gerar restante por crossover e mutação
            while len(next_generation) < self.population_size:
                # Seleção por torneio
                tournament_size = 5
                tournament = random.sample(fitness_scores, tournament_size)
                parent1 = max(tournament, key=lambda x: x[1])[0]
                
                tournament = random.sample(fitness_scores, tournament_size)
                parent2 = max(tournament, key=lambda x: x[1])[0]
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutação
                child = self.mutate(child)
                
                next_generation.append(child)
            
            population = next_generation
        
        # Estatísticas finais
        total_price = sum(a['preco'] for a in best_team)
        total_points = sum(a['predicao'] for a in best_team)
        
        stats = {
            'total_pontos_preditos': total_points,
            'total_preco': total_price,
            'patrimonio_usado': (total_price / self.patrimonio) * 100,
            'fitness': best_fitness,
            'formacao': self.formacao_nome
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"OTIMIZAÇÃO CONCLUÍDA")
        logger.info(f"Pontos Preditos: {total_points:.2f}")
        logger.info(f"Preço Total: C$ {total_price:.2f}")
        logger.info(f"Patrimônio Usado: {stats['patrimonio_usado']:.1f}%")
        logger.info(f"{'='*50}\n")
        
        return best_team, stats
    
    def format_team_output(self, team: List[Dict]) -> pd.DataFrame:
        """Formata time otimizado para exibição"""
        df = pd.DataFrame(team)
        
        df['posicao_nome'] = df['posicao_id'].map(POSICOES)
        
        # Sort first, then slice
        df = df.sort_values(['posicao_id', 'predicao'], ascending=[True, False])
        df = df[['apelido', 'posicao_nome', 'clube_id', 'preco', 'predicao', 'media']]
        
        df.columns = ['Atleta', 'Posição', 'Clube', 'Preço (C$)', 'Predição', 'Média']
        
        return df
