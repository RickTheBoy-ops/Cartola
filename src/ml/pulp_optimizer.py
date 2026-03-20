"""
Otimizador de Escalação via Programação Linear Inteira (PuLP)

Garante a escalação MATEMATICAMENTE ÓTIMA para o orçamento disponível.
Usado em conjunto com o otimizador genético para geração de escalação consolidada.

Instalação:
    pip install pulp
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Mapeamento posicao_id -> chave interna
POSICOES_ID = {
    1: 'GOL',
    2: 'LAT',
    3: 'ZAG',
    4: 'MEI',
    5: 'ATA',
    6: 'TEC',
}

# Formações suportadas: qtd de cada posição [GOL, ZAG, LAT, MEI, ATA, TEC]
FORMACOES_PULP: Dict[str, Dict[str, int]] = {
    '3-4-3': {'GOL': 1, 'ZAG': 3, 'LAT': 0, 'MEI': 4, 'ATA': 3, 'TEC': 1},
    '3-5-2': {'GOL': 1, 'ZAG': 3, 'LAT': 0, 'MEI': 5, 'ATA': 2, 'TEC': 1},
    '4-3-3': {'GOL': 1, 'ZAG': 2, 'LAT': 2, 'MEI': 3, 'ATA': 3, 'TEC': 1},
    '4-4-2': {'GOL': 1, 'ZAG': 2, 'LAT': 2, 'MEI': 4, 'ATA': 2, 'TEC': 1},
    '4-5-1': {'GOL': 1, 'ZAG': 2, 'LAT': 2, 'MEI': 5, 'ATA': 1, 'TEC': 1},
    '5-3-2': {'GOL': 1, 'ZAG': 3, 'LAT': 2, 'MEI': 3, 'ATA': 2, 'TEC': 1},
    '5-4-1': {'GOL': 1, 'ZAG': 3, 'LAT': 2, 'MEI': 4, 'ATA': 1, 'TEC': 1},
}


class PuLPOptimizer:
    """
    Otimizador de Programação Linear para escalação do Cartola FC.

    Recebe o DataFrame de atletas com `score_cruzado` (gerado pelo pipeline
    do main.py) e encontra a combinação MATEMATICAMENTE ÓTIMA.

    Parâmetros
    ----------
    atletas_df : pd.DataFrame
        Atletas disponíveis com colunas:
        atleta_id, apelido, posicao_id, clube_id, preco, score_cruzado
    patrimonio : float
        Orçamento máximo em cartoletas
    formacao : str
        Formação tática (ex: '4-3-3')
    max_mesmo_clube : int
        Máximo de atletas do mesmo clube (padrão 3)
    score_col : str
        Coluna usada como objetivo de maximização
    """

    def __init__(
        self,
        atletas_df: pd.DataFrame,
        patrimonio: float,
        formacao: str = '4-3-3',
        max_mesmo_clube: int = 3,
        score_col: str = 'score_cruzado',
    ):
        try:
            import pulp  # noqa: F401
        except ImportError:
            raise ImportError(
                "PuLP não instalado. Execute: pip install pulp"
            )

        self.patrimonio      = patrimonio
        self.formacao_nome   = formacao
        self.max_mesmo_clube = max_mesmo_clube
        self.score_col       = score_col

        if formacao not in FORMACOES_PULP:
            logger.warning(f"⚠️  Formação '{formacao}' desconhecida. Usando 4-3-3.")
            self.formacao_nome = '4-3-3'
        self.formacao = FORMACOES_PULP[self.formacao_nome]

        # Garantir colunas mínimas
        df = atletas_df.copy()
        if score_col not in df.columns:
            fallback = next(
                (c for c in ['score_final', 'predicao_ajustada', 'predicao'] if c in df.columns),
                None
            )
            if fallback:
                logger.warning(
                    f"⚠️  '{score_col}' não encontrado. Usando '{fallback}' como score."
                )
                df[score_col] = df[fallback]
            else:
                df[score_col] = df.get('media', 0)

        # Mapear posicao_id -> chave string
        df['pos_key'] = df['posicao_id'].map(POSICOES_ID)
        df = df.dropna(subset=['pos_key', 'preco', score_col])
        df = df[df['preco'] > 0]
        df = df.reset_index(drop=True)

        self.df      = df
        self.indices = list(df.index)

        logger.info(
            f"📊 PuLP Otimizador | Formação: {self.formacao_nome} | "
            f"Orçamento: C${patrimonio:.2f} | Atletas: {len(df)}"
        )

    def optimize(self) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Executa a otimização linear e retorna:
        - DataFrame com os atletas escalados
        - Dicionário com estatísticas
        """
        import pulp

        prob = pulp.LpProblem("Cartola_PuLP", pulp.LpMaximize)

        # Variáveis binárias: 1 = escalado, 0 = não escalado
        x = pulp.LpVariable.dicts("atleta", self.indices, cat="Binary")

        # ========================================================
        # OBJETIVO: maximizar soma dos scores cruzados
        # ========================================================
        prob += pulp.lpSum(
            self.df.loc[i, self.score_col] * x[i]
            for i in self.indices
        ), "Maximizar_Score_Cruzado"

        # ========================================================
        # RESTRIÇÃO 1: Orçamento
        # ========================================================
        prob += (
            pulp.lpSum(self.df.loc[i, 'preco'] * x[i] for i in self.indices)
            <= self.patrimonio
        ), "Orcamento"

        # ========================================================
        # RESTRIÇÃO 2: Quantidade exata por posição (formação)
        # ========================================================
        for pos_key, qtd in self.formacao.items():
            idx_pos = [i for i in self.indices if self.df.loc[i, 'pos_key'] == pos_key]
            prob += (
                pulp.lpSum(x[i] for i in idx_pos) == qtd
            ), f"Qtd_{pos_key}"

        # ========================================================
        # RESTRIÇÃO 3: Máx. atletas do mesmo clube
        # ========================================================
        clubes = self.df['clube_id'].unique()
        for clube in clubes:
            idx_clube = [i for i in self.indices if self.df.loc[i, 'clube_id'] == clube]
            prob += (
                pulp.lpSum(x[i] for i in idx_clube) <= self.max_mesmo_clube
            ), f"MaxClube_{int(clube)}"

        # ========================================================
        # RESTRIÇÃO 4 (opcional): Máx. 2 defensores do mesmo clube
        # (evita perder SG coletivo por gol sofrido)
        # ========================================================
        pos_defesa = {'GOL', 'ZAG', 'LAT'}
        for clube in clubes:
            idx_def = [
                i for i in self.indices
                if self.df.loc[i, 'clube_id'] == clube
                and self.df.loc[i, 'pos_key'] in pos_defesa
            ]
            if idx_def:
                prob += (
                    pulp.lpSum(x[i] for i in idx_def) <= 2
                ), f"MaxDefClube_{int(clube)}"

        # ========================================================
        # RESOLVER
        # ========================================================
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]

        if status != 'Optimal':
            logger.error(
                f"❌ PuLP não encontrou solução ótima. Status: {status}. "
                f"Verifique orçamento e formação."
            )
            return None, {'status': status, 'total_preco': 0, 'total_score': 0}

        # ========================================================
        # EXTRAIR TIME
        # ========================================================
        escalados = [
            i for i in self.indices
            if pulp.value(x[i]) is not None and round(pulp.value(x[i])) == 1
        ]
        team_df = self.df.loc[escalados].copy()

        total_preco = team_df['preco'].sum()
        total_score = team_df[self.score_col].sum()

        # Score base (sem mult confronto) para exibição
        score_base_col = next(
            (c for c in ['score_final', 'predicao_ajustada', 'predicao']
             if c in team_df.columns),
            self.score_col
        )
        total_pts_pred = team_df.get(score_base_col, team_df[self.score_col]).sum()

        stats = {
            'status':               'Optimal',
            'total_preco':          round(total_preco, 2),
            'total_score_cruzado':  round(total_score, 2),
            'total_pontos_preditos': round(total_pts_pred, 2),
            'patrimonio_usado_pct': round((total_preco / self.patrimonio) * 100, 1),
            'formacao':             self.formacao_nome,
            'n_atletas':            len(team_df),
        }

        logger.info(
            f"✅ PuLP Ótimo | Score: {total_score:.2f} | "
            f"Preço: C${total_preco:.2f} ({stats['patrimonio_usado_pct']}%) | "
            f"{len(team_df)} atletas"
        )
        return team_df, stats

    @staticmethod
    def disponivel() -> bool:
        """Verifica se PuLP está instalado."""
        try:
            import pulp  # noqa: F401
            return True
        except ImportError:
            return False
