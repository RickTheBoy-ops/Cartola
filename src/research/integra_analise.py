"""
Integração entre web_research e AnalisadorEspecialista

Fluxo:
  1. Recebe lista de jogadores candidatos + lista de times
  2. Pesquisa todos em paralelo (10+ fontes cada)
  3. Injeta contexto web no prompt da análise LLM
  4. Enriquece os objetos Jogador com dados extraídos
"""

import logging
from typing import List, Dict, Any, Optional

from src.analise_especialista import Jogador, Rodada, AnalisadorEspecialista
from src.research.web_research import (
    pesquisar_jogador,
    pesquisar_time,
    pesquisar_rodada_completa,
    PesquisaJogador,
    PesquisaTime,
)

logger = logging.getLogger(__name__)


def enriquecer_jogadores_com_web(
    jogadores: List[Jogador],
    times_rodada: List[str],
    forcar_atualizacao: bool = False,
) -> Dict[str, Any]:
    """
    Recebe lista de Jogador (modelo interno) e pesquisa cada um na web.

    Returns:
        {
            "pesquisas":         {nome: PesquisaJogador},
            "pesquisas_times":   {time: PesquisaTime},
            "contexto_llm":      str,  <- pronto para injetar no prompt
            "noticias_desfalques": [str]  <- trechos de notícias sobre desfalques
        }
    """
    payload_jogadores = [
        {"nome": j.nome, "time": j.time}
        for j in jogadores
    ]

    resultado = pesquisar_rodada_completa(
        jogadores=payload_jogadores,
        times=times_rodada,
    )

    # Extração simples de trechos sobre desfalques
    noticias_desfalques = []
    for nome, pj in resultado["jogadores"].items():
        for r in pj.resultados:
            if r.status == "ok" and any(
                palavra in r.conteudo_bruto.lower()
                for palavra in ["lesao", "lesão", "desfalque", "suspenso", "contundido", "dúvida"]
            ):
                noticias_desfalques.append(
                    f"{nome}: [{r.fonte}] {r.conteudo_bruto[:200]}"
                )

    resultado["noticias_desfalques"] = noticias_desfalques
    return resultado


def construir_prompt_com_contexto_web(
    prompt_base: str,
    contexto_web: str,
    noticias_desfalques: List[str],
) -> str:
    """
    Injeta o contexto de pesquisa web no prompt base da análise LLM.
    """
    bloco_desfalques = ""
    if noticias_desfalques:
        bloco_desfalques = "\n".join(
            f"  - {n}" for n in noticias_desfalques[:15]
        )
        bloco_desfalques = f"\n\n### ALERTAS DE DESFALQUES (extraídos da web):\n{bloco_desfalques}"

    return (
        f"{prompt_base}"
        f"\n\n{'='*60}"
        f"\n### CONTEXTO EXTRAÍDO DA WEB (12 fontes por jogador/time):"
        f"\n{'='*60}\n"
        f"{contexto_web[:12000]}"  # limita para não estourar janela de contexto
        f"{bloco_desfalques}"
        f"\n\nUse TODAS as informações acima para fundamentar sua análise."
    )


class AnalisadorComPesquisaWeb(AnalisadorEspecialista):
    """
    Extensão do AnalisadorEspecialista que realiza pesquisa web
    antes de cada análise, enriquecendo o contexto do LLM.
    """

    def __init__(self, perplexity_client=None):
        super().__init__()
        self._perplexity_client = perplexity_client
        self._contexto_web: str = ""
        self._noticias_desfalques: List[str] = []

    def executar_analise_completa_com_web(
        self,
        rodada: Rodada,
        estrategia: str = "meio-termo",
        forcar_pesquisa: bool = False,
    ) -> Dict[str, Any]:
        """
        Pipeline completo:
          1. Pesquisa web (10+ fontes por jogador/time)
          2. Análise especialista enriquecida
          3. Retorna resultado + contexto web + alertas
        """
        # Etapa 0: Pesquisa web
        logger.info(f"[Rodada {rodada.numero}] Iniciando pesquisa web em paralelo...")
        times_unicos = list(set(j.time for j in rodada.jogadores))

        pesquisa_resultado = enriquecer_jogadores_com_web(
            jogadores=rodada.jogadores,
            times_rodada=times_unicos,
            forcar_atualizacao=forcar_pesquisa,
        )

        self._contexto_web = pesquisa_resultado["contexto_completo"]
        self._noticias_desfalques = pesquisa_resultado.get("noticias_desfalques", [])

        logger.info(
            f"Pesquisa web concluída: "
            f"{len(pesquisa_resultado['jogadores'])} jogadores, "
            f"{len(pesquisa_resultado['times'])} times | "
            f"{len(self._noticias_desfalques)} alertas de desfalque"
        )

        # Etapa 1–6: Análise especialista normal
        resultado = self.executar_analise_completa(rodada, estrategia)

        # Adiciona dados da web ao resultado
        resultado["pesquisa_web"] = {
            "fontes_por_jogador": {
                nome: {
                    "fontes_consultadas": pj.fontes_consultadas,
                    "fontes_sucesso": pj.fontes_sucesso,
                }
                for nome, pj in pesquisa_resultado["jogadores"].items()
            },
            "alertas_desfalques": self._noticias_desfalques,
            "contexto_chars": len(self._contexto_web),
        }

        return resultado

    def gerar_relatorio_markdown(self, rodada: Rodada) -> str:
        """Inclui secção de alertas web no relatório"""
        md = super().gerar_relatorio_markdown(rodada)

        if self._noticias_desfalques:
            md += "\n\n---\n\n## ⚠️ ALERTAS DE DESFALQUE (Pesquisa Web)\n\n"
            for alerta in self._noticias_desfalques[:10]:
                md += f"- {alerta}\n"

        if self._contexto_web:
            md += (
                f"\n\n---\n\n## 🔍 Pesquisa Web\n\n"
                f"Contexto extraído de **12 fontes** (Sofascore, FBref, Flashscore, "
                f"Whoscored, GE Globo, ESPN, Cartoleiro, RedaNão, Terrão, "
                f"Cartola API, Transfermarkt, Google News).  \n"
                f"Total de caracteres extraídos: **{len(self._contexto_web):,}**\n"
            )

        return md


# ---------------------------------------------------------------------------
# Uso rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Teste rápido sem banco de dados
    from src.analise_especialista import Jogador, Rodada, Confronto, TipoConfrontoEnum
    from datetime import datetime

    jogadores_mock = [
        Jogador(1, "Hulk", "Atlético-MG", "atacante", 18.0, 12.5, 12.5, 20.0, "BAIXO", 0.9, 40, 15.0, 1.8),
        Jogador(2, "Cano", "Fluminense", "atacante", 16.0, 10.0, 10.0, 18.0, "BAIXO", 0.85, 30, 12.0, 1.5),
    ]
    confrontos_mock = [
        Confronto(1, "Atlético-MG", "Fluminense", True, 0.65, TipoConfrontoEnum.GRUPO_A, [], [], 2.5),
    ]
    rodada_mock = Rodada(
        numero=10,
        data=datetime.now().strftime("%d/%m/%Y"),
        confrontos=confrontos_mock,
        jogadores=jogadores_mock,
        patrimonio_total=100.0,
        patrimonio_livre_apos_11=100.0,
    )

    analisador = AnalisadorComPesquisaWeb()
    resultado = analisador.executar_analise_completa_com_web(rodada_mock)

    print("\n=== RESULTADO ===")
    print(f"Time: {[j.nome for j in resultado['time']]}")
    print(f"Cap: {resultado['capitao'].nome}")
    print(f"Alertas web: {len(resultado['pesquisa_web']['alertas_desfalques'])}")
    print(resultado["relatorio"][:1000])
