"""
Agente de Análise de Notícias de Última Hora para Cartola FC.
Busca notícias recentes de jogadores, classifica impacto na escalação
e retorna score de risco/oportunidade por jogador.
"""

from __future__ import annotations

import os
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

RSS_FEEDS = [
    "https://ge.globo.com/rss/feed.xml",
    "https://www.espnbrasil.com.br/rss/feed.xml",
    "https://www.uol.com.br/esporte/futebol/rss.xml",
]

# Palavras-chave que impactam positivamente a escalação
KEYWORDS_POSITIVO = [
    "retorno", "recuperado", "treinando", "titular",
    "convocado", "seleção", "destaque", "gol", "assistência",
    "melhor", "artilheiro",
]

# Palavras-chave que impactam negativamente
KEYWORDS_NEGATIVO = [
    "lesão", "lesionado", "contundido", "suspensão", "suspenso",
    "expulso", "cartão vermelho", "dúvida", "desfalque",
    "cirurgia", "afastado", "corte", "covid", "mal-estar",
]


# ---------------------------------------------------------------------------
# Modelos de dados
# ---------------------------------------------------------------------------

class NoticiaJogador:
    """Representa uma notícia associada a um jogador."""

    def __init__(
        self,
        jogador: str,
        titulo: str,
        resumo: str,
        fonte: str,
        data: datetime,
        impacto: str,
        score: float,
        url: str = "",
    ) -> None:
        self.jogador = jogador
        self.titulo = titulo
        self.resumo = resumo
        self.fonte = fonte
        self.data = data
        self.impacto = impacto   # 'positivo' | 'negativo' | 'neutro'
        self.score = score       # -1.0 a +1.0
        self.url = url

    def to_dict(self) -> dict:
        return {
            "jogador": self.jogador,
            "titulo": self.titulo,
            "resumo": self.resumo,
            "fonte": self.fonte,
            "data": self.data.isoformat(),
            "impacto": self.impacto,
            "score": self.score,
            "url": self.url,
        }


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _classificar_impacto(texto: str) -> tuple[str, float]:
    """Classifica o impacto de um texto como positivo, negativo ou neutro.

    Retorna uma tupla (impacto: str, score: float) onde score vai de -1.0 a +1.0.
    """
    texto_lower = texto.lower()

    pontos_pos = sum(1 for kw in KEYWORDS_POSITIVO if kw in texto_lower)
    pontos_neg = sum(1 for kw in KEYWORDS_NEGATIVO if kw in texto_lower)

    total = pontos_pos + pontos_neg
    if total == 0:
        return "neutro", 0.0

    score_bruto = (pontos_pos - pontos_neg) / total

    if score_bruto > 0.2:
        return "positivo", round(score_bruto, 2)
    elif score_bruto < -0.2:
        return "negativo", round(score_bruto, 2)
    else:
        return "neutro", round(score_bruto, 2)


def _menciona_jogador(texto: str, nome_jogador: str) -> bool:
    """Verifica se o texto menciona o jogador (busca case-insensitive por partes do nome)."""
    partes = nome_jogador.lower().split()
    texto_lower = texto.lower()
    return any(parte in texto_lower for parte in partes if len(parte) > 3)


# ---------------------------------------------------------------------------
# Busca via RSS
# ---------------------------------------------------------------------------

def buscar_noticias_rss(
    jogadores: list[str],
    horas: int = 24,
) -> list[NoticiaJogador]:
    """Busca notícias nos feeds RSS configurados e filtra por jogadores.

    Args:
        jogadores: Lista de nomes de jogadores a monitorar.
        horas: Janela temporal em horas (padrão 24h).

    Returns:
        Lista de NoticiaJogador com as notícias encontradas.
    """
    noticias: list[NoticiaJogador] = []
    limite_data = datetime.utcnow() - timedelta(hours=horas)

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.utcnow()
                if pub_date < limite_data:
                    continue

                titulo = getattr(entry, "title", "")
                resumo = getattr(entry, "summary", "")
                texto_completo = f"{titulo} {resumo}"

                for jogador in jogadores:
                    if _menciona_jogador(texto_completo, jogador):
                        impacto, score = _classificar_impacto(texto_completo)
                        noticias.append(
                            NoticiaJogador(
                                jogador=jogador,
                                titulo=titulo,
                                resumo=resumo[:300],
                                fonte=feed.feed.get("title", feed_url),
                                data=pub_date,
                                impacto=impacto,
                                score=score,
                                url=getattr(entry, "link", ""),
                            )
                        )
        except Exception as exc:
            logger.warning("Erro ao buscar feed %s: %s", feed_url, exc)

    return noticias


# ---------------------------------------------------------------------------
# Busca via NewsAPI
# ---------------------------------------------------------------------------

def buscar_noticias_newsapi(
    jogadores: list[str],
    horas: int = 24,
) -> list[NoticiaJogador]:
    """Busca notícias via NewsAPI para cada jogador.

    Requer NEWS_API_KEY configurada no .env.

    Args:
        jogadores: Lista de nomes de jogadores.
        horas: Janela temporal em horas.

    Returns:
        Lista de NoticiaJogador.
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY não configurada. Pulando NewsAPI.")
        return []

    noticias: list[NoticiaJogador] = []
    desde = (datetime.utcnow() - timedelta(hours=horas)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for jogador in jogadores:
        try:
            resp = requests.get(
                NEWS_API_URL,
                params={
                    "q": f'{jogador} futebol OR Cartola',
                    "language": "pt",
                    "from": desde,
                    "sortBy": "publishedAt",
                    "apiKey": NEWS_API_KEY,
                    "pageSize": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for article in data.get("articles", []):
                titulo = article.get("title", "")
                resumo = article.get("description", "") or ""
                texto_completo = f"{titulo} {resumo}"

                if not _menciona_jogador(texto_completo, jogador):
                    continue

                impacto, score = _classificar_impacto(texto_completo)
                pub_date_str = article.get("publishedAt", "")
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                except Exception:
                    pub_date = datetime.utcnow()

                noticias.append(
                    NoticiaJogador(
                        jogador=jogador,
                        titulo=titulo,
                        resumo=resumo[:300],
                        fonte=article.get("source", {}).get("name", "NewsAPI"),
                        data=pub_date,
                        impacto=impacto,
                        score=score,
                        url=article.get("url", ""),
                    )
                )
        except Exception as exc:
            logger.warning("Erro NewsAPI para '%s': %s", jogador, exc)

    return noticias


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------

def analisar_noticias_jogadores(
    jogadores: list[str],
    horas: int = 24,
) -> dict[str, dict]:
    """Ponto de entrada principal do agente de notícias.

    Agrega notícias de todas as fontes e retorna um resumo por jogador
    com score final de risco/oportunidade.

    Args:
        jogadores: Lista de nomes de jogadores.
        horas: Janela temporal em horas para busca de notícias.

    Returns:
        Dicionário {nome_jogador: {score, impacto, noticias: [...]}}
    """
    todas_noticias: list[NoticiaJogador] = []
    todas_noticias.extend(buscar_noticias_rss(jogadores, horas))
    todas_noticias.extend(buscar_noticias_newsapi(jogadores, horas))

    resultado: dict[str, dict] = {}

    for jogador in jogadores:
        noticias_jogador = [n for n in todas_noticias if n.jogador == jogador]

        if not noticias_jogador:
            resultado[jogador] = {
                "score": 0.0,
                "impacto": "neutro",
                "total_noticias": 0,
                "noticias": [],
            }
            continue

        score_medio = sum(n.score for n in noticias_jogador) / len(noticias_jogador)
        score_medio = round(score_medio, 2)

        if score_medio > 0.1:
            impacto_geral = "positivo"
        elif score_medio < -0.1:
            impacto_geral = "negativo"
        else:
            impacto_geral = "neutro"

        resultado[jogador] = {
            "score": score_medio,
            "impacto": impacto_geral,
            "total_noticias": len(noticias_jogador),
            "noticias": [n.to_dict() for n in sorted(noticias_jogador, key=lambda x: x.data, reverse=True)],
        }

    return resultado


if __name__ == "__main__":
    import json
    jogadores_teste = ["Hulk", "Pedro", "Arrascaeta", "Raphinha", "Vini Jr"]
    resultado = analisar_noticias_jogadores(jogadores_teste, horas=48)
    print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
