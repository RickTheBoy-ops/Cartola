#!/usr/bin/env python3
"""Utilitário de linha de comando para avaliar performance de escalações.

Exemplos de uso:

    python scripts/avaliar_escalacoes.py --ano 2026 --rodada 10
    python scripts/avaliar_escalacoes.py --ano 2026 --historico 10
"""

import argparse
from pathlib import Path

from src.analysis.performance import calcular_performance_rodada, historico_performance


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia performance das escalações otimizadas (sem IA)")
    parser.add_argument("--db", default="data/cartola.db", help="Caminho para o banco SQLite")
    parser.add_argument("--ano", type=int, default=2026, help="Ano da temporada a analisar")
    parser.add_argument("--rodada", type=int, help="Rodada específica para avaliar")
    parser.add_argument("--historico", type=int, help="Quantidade de últimas rodadas para análise histórica")

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.rodada is not None:
        df = calcular_performance_rodada(db_path, ano=args.ano, rodada=args.rodada)
        if df.empty:
            print(f"Nenhuma escalação encontrada para ano={args.ano}, rodada={args.rodada}.")
        else:
            print("=== Performance da Rodada ===")
            print(df.to_string(index=False))

    if args.historico is not None:
        df_hist = historico_performance(db_path, ano=args.ano, ultimas_n_rodadas=args.historico)
        if df_hist.empty:
            print(f"Nenhuma escalação histórica encontrada para ano={args.ano}.")
        else:
            print("\n=== Histórico de Performance ===")
            print(df_hist.to_string(index=False))


if __name__ == "__main__":
    main()
