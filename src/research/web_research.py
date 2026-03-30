"""
Módulo de Pesquisa Web Multi-Fonte — Cartola FC

Raspa pelo menos 10 fontes púr jogador/time antes de montar a análise.

Fontes monitoradas:
  1. Sofascore        — estatísticas detalhadas do jogador
  2. FBref            — dados avançados (xG, xA, pressões)
  3. Flashscore       — forma recente + resultados
  4. Whoscored        — rating e heatmap
  5. GE Esportes (ge.globo.com) — notícias, desfalques, entrevistas
  6. TNT Sports / ESPN Brasil   — escalacões prováveis
  7. Cartoleiro.com   — dicas especializadas Cartola
  8. RedaNão         — análise de mercado Cartola
  9. Terrão do Cartola — parciais de scouts e variação de preço
 10. Mapeião / CartolaPedia — histórico de pontuações Cartola
 11. Transfermarkt   — valor de mercado + histórico de contusões
 12. Google News      — últimas notícias gerais sobre o jogador/time
"""

import time
import logging
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SOURCE_URLS: Dict[str, str] = {
    "sofascore_player":    "https://api.sofascore.com/api/v1/player/{player_id}/statistics/season/{season_id}",
    "sofascore_team":      "https://api.sofascore.com/api/v1/team/{team_id}/statistics/season/{season_id}",
    "fbref_player":        "https://fbref.com/pt/jogadores/{fbref_id}/",
    "flashscore_player":   "https://www.flashscore.com.br/jogador/{slug}/",
    "whoscored_player":    "https://www.whoscored.com/Players/{ws_id}/",
    "ge_noticias_jogador": "https://ge.globo.com/busca/?q={nome_jogador}+Brasileir%C3%A3o",
    "ge_noticias_time":    "https://ge.globo.com/futebol/times/{slug_time}/",
    "espn_escalacao":      "https://www.espn.com.br/futebol/time/_/id/{espn_id}/",
    "cartoleiro":          "https://www.cartoleiro.com.br/jogador/{slug}",
    "redanao":             "https://www.redanao.com.br/jogador/{slug}",
    "terrao":              "https://terraodocartola.com.br/jogador/{nome_jogador}",
    "transfermarkt":       "https://www.transfermarkt.com.br/jogador/{slug}/profil/spieler/{tm_id}",
    "google_news":         "https://news.google.com/rss/search?q={query}+futebol+brasileirao&hl=pt-BR&gl=BR&ceid=BR:pt-419",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SOFASCORE_HEADERS = {
    **DEFAULT_HEADERS,
    "referer": "https://www.sofascore.com/",
}

REQUEST_TIMEOUT = 10  # segundos
MAX_WORKERS = 6       # paralelismo
CACHE_TTL_MINUTES = 30


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class FonteResultado:
    """Resultado bruto de uma fonte de pesquisa"""
    fonte: str
    url: str
    status: str                  # "ok" | "erro" | "timeout" | "vazio"
    dados: Dict[str, Any]        # dados extraidos
    conteudo_bruto: str = ""     # texto extraído para o LLM processar
    tempo_ms: float = 0.0
    erro: str = ""


@dataclass
class PesquisaJogador:
    """Resultado consolidado de pesquisa sobre um jogador"""
    nome: str
    time: str
    fontes_consultadas: int
    fontes_sucesso: int
    resultados: List[FonteResultado] = field(default_factory=list)
    dados_consolidados: Dict[str, Any] = field(default_factory=dict)
    gerado_em: str = ""

    def to_context_string(self) -> str:
        """Gera string de contexto para enviar ao LLM"""
        linhas = [
            f"=== PESQUISA WEB: {self.nome} ({self.time}) ===",
            f"Fontes consultadas: {self.fontes_consultadas} | Sucesso: {self.fontes_sucesso}",
            "",
        ]
        for r in self.resultados:
            if r.status == "ok" and r.conteudo_bruto:
                linhas.append(f"--- [{r.fonte.upper()}] ({r.url}) ---")
                # Limita a 800 chars por fonte para não explodir o contexto
                linhas.append(r.conteudo_bruto[:800])
                linhas.append("")
        return "\n".join(linhas)


@dataclass
class PesquisaTime:
    """Resultado consolidado de pesquisa sobre um time"""
    nome: str
    fontes_consultadas: int
    fontes_sucesso: int
    resultados: List[FonteResultado] = field(default_factory=list)
    dados_consolidados: Dict[str, Any] = field(default_factory=dict)
    gerado_em: str = ""

    def to_context_string(self) -> str:
        linhas = [
            f"=== PESQUISA WEB TIME: {self.nome} ===",
            f"Fontes consultadas: {self.fontes_consultadas} | Sucesso: {self.fontes_sucesso}",
            "",
        ]
        for r in self.resultados:
            if r.status == "ok" and r.conteudo_bruto:
                linhas.append(f"--- [{r.fonte.upper()}] ({r.url}) ---")
                linhas.append(r.conteudo_bruto[:600])
                linhas.append("")
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Cache simples em memória (evita raspar a mesma página várias vezes)
# ---------------------------------------------------------------------------

class _MemCache:
    def __init__(self, ttl_minutes: int = CACHE_TTL_MINUTES):
        self._store: Dict[str, tuple] = {}  # key → (timestamp, data)
        self._ttl = timedelta(minutes=ttl_minutes)

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            ts, data = self._store[key]
            if datetime.now() - ts < self._ttl:
                return data
        return None

    def set(self, key: str, data: Any):
        self._store[key] = (datetime.now(), data)


_cache = _MemCache()


# ---------------------------------------------------------------------------
# Helpers de requisição
# ---------------------------------------------------------------------------

def _get(url: str, headers: Optional[Dict] = None, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """GET com retry simples e tratamento de erros"""
    hdrs = headers or DEFAULT_HEADERS
    for tentativa in range(2):
        try:
            resp = requests.get(url, headers=hdrs, timeout=timeout)
            if resp.status_code == 200:
                return resp
            logger.debug(f"HTTP {resp.status_code} em {url}")
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout em {url} (tentativa {tentativa + 1})")
        except requests.exceptions.RequestException as exc:
            logger.debug(f"Erro na requisição {url}: {exc}")
            break
        time.sleep(0.5)
    return None


def _text_from_soup(soup: BeautifulSoup, seletores: List[str]) -> str:
    """Extrai texto dos primeiros elementos encontrados"""
    partes = []
    for sel in seletores:
        els = soup.select(sel)[:5]
        for el in els:
            texto = el.get_text(separator=" ", strip=True)
            if texto:
                partes.append(texto)
    return " | ".join(partes)


# ---------------------------------------------------------------------------
# Scraper por fonte
# ---------------------------------------------------------------------------

def _raspar_sofascore_jogador(nome: str, sofascore_id: Optional[int] = None) -> FonteResultado:
    """Busca rating + estatísticas no Sofascore via API pública"""
    url = f"https://api.sofascore.com/api/v1/player/{sofascore_id or 0}/characteristics"
    inicio = time.time()
    if not sofascore_id:
        # Busca textual
        url_busca = f"https://api.sofascore.com/api/v1/search/player-team-unique-tournament/?q={nome.replace(' ', '+')}&sport=football"
        resp = _get(url_busca, SOFASCORE_HEADERS)
        if resp:
            dados = resp.json()
            players = dados.get("players", [])
            if players:
                sofascore_id = players[0].get("id")
                nome_real = players[0].get("name", nome)
                return FonteResultado(
                    fonte="sofascore",
                    url=url_busca,
                    status="ok",
                    dados={"sofascore_id": sofascore_id, "nome": nome_real, "dados_raw": players[0]},
                    conteudo_bruto=f"Sofascore: {nome_real} | ID {sofascore_id} | {json.dumps(players[0])[:500]}",
                    tempo_ms=(time.time() - inicio) * 1000,
                )
        return FonteResultado(fonte="sofascore", url=url_busca, status="vazio",
                              dados={}, tempo_ms=(time.time() - inicio) * 1000)

    resp = _get(url, SOFASCORE_HEADERS)
    if resp:
        dados = resp.json()
        return FonteResultado(
            fonte="sofascore",
            url=url,
            status="ok",
            dados=dados,
            conteudo_bruto=f"Sofascore stats: {json.dumps(dados)[:800]}",
            tempo_ms=(time.time() - inicio) * 1000,
        )
    return FonteResultado(fonte="sofascore", url=url, status="erro", dados={},
                          tempo_ms=(time.time() - inicio) * 1000)


def _raspar_ge_noticias(query: str, tipo: str = "jogador") -> FonteResultado:
    """Busca notícias recentes no GE Globo"""
    url = f"https://ge.globo.com/busca/?q={query.replace(' ', '+')}&species=noticia"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="ge_globo", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    titulos = [el.get_text(strip=True) for el in soup.select(".widget--info__title")[:6]]
    descricoes = [el.get_text(strip=True) for el in soup.select(".widget--info__description")[:6]]
    conteudo = " | ".join(titulos + descricoes)
    return FonteResultado(
        fonte="ge_globo",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={"titulos": titulos},
        conteudo_bruto=conteudo[:800],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_google_news_rss(query: str) -> FonteResultado:
    """Busca RSS do Google News para o jogador/time"""
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+').replace('/', '')}+futebol+brasileirao&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="google_news", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")[:8]
    noticias = []
    for item in items:
        titulo = item.find("title")
        pub = item.find("pubDate")
        noticias.append({
            "titulo": titulo.text if titulo else "",
            "data": pub.text if pub else "",
        })
    conteudo = " | ".join(n["titulo"] for n in noticias)
    return FonteResultado(
        fonte="google_news",
        url=url,
        status="ok" if noticias else "vazio",
        dados={"noticias": noticias},
        conteudo_bruto=conteudo[:800],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_transfermarkt(nome: str, slug: str = "") -> FonteResultado:
    """Raspa valor de mercado e histórico de contusões do Transfermarkt"""
    query = slug or nome.lower().replace(" ", "-")
    url = f"https://www.transfermarkt.com.br/schnellsuche/ergebnis/schnellsuche?query={query.replace('-', '+')}&Spieler_page=0"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="transfermarkt", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    nomes = [el.get_text(strip=True) for el in soup.select(".hauptlink a")[:3]]
    valores = [el.get_text(strip=True) for el in soup.select(".rechts.hauptlink a")[:3]]
    conteudo = f"Jogadores encontrados: {', '.join(nomes)} | Valores: {', '.join(valores)}"
    return FonteResultado(
        fonte="transfermarkt",
        url=url,
        status="ok" if nomes else "vazio",
        dados={"nomes": nomes, "valores": valores},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_flashscore(nome: str) -> FonteResultado:
    """Busca forma recente e resultados no Flashscore"""
    slug = nome.lower().replace(" ", "-")
    url = f"https://www.flashscore.com.br/jogador/{slug}/"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="flashscore", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, [
        ".participant__participantName",
        ".event__score",
        ".event__participant"
    ])
    return FonteResultado(
        fonte="flashscore",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_cartoleiro(nome: str) -> FonteResultado:
    """Dicas especializadas Cartola no Cartoleiro.com.br"""
    slug = nome.lower().replace(" ", "-")
    url = f"https://www.cartoleiro.com.br/jogador/{slug}"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="cartoleiro", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, [
        ".player-stats", ".player-info", ".player-history",
        "h1", ".stats-table"
    ])
    return FonteResultado(
        fonte="cartoleiro",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_redanao(nome: str) -> FonteResultado:
    """Análise de mercado Cartola no Reda Não"""
    slug = nome.lower().replace(" ", "-")
    url = f"https://www.redanao.com.br/jogador/{slug}"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="redanao", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, [".player", ".stats", "h1", "table"])
    return FonteResultado(
        fonte="redanao",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_terrao(nome: str) -> FonteResultado:
    """Parciais de scouts e variação de preço no Terrão do Cartola"""
    query = nome.replace(" ", "+")
    url = f"https://terraodocartola.com.br/?s={query}"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="terrao_cartola", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, ["h2.entry-title", ".entry-summary", "article"])
    return FonteResultado(
        fonte="terrao_cartola",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_cartola_api_oficial(slug_nome: str) -> FonteResultado:
    """API pública do próprio Cartola FC (busca atletica)"""
    url = f"https://api.cartolafc.globo.com/atletas/busca?q={slug_nome.replace(' ', '+')}"
    inicio = time.time()
    resp = _get(url, headers={**DEFAULT_HEADERS, "referer": "https://cartolafc.globo.com/"})
    if not resp:
        return FonteResultado(fonte="cartola_api", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    try:
        dados = resp.json()
        atletas = dados.get("atletas", dados) if isinstance(dados, dict) else dados
        if isinstance(atletas, list) and atletas:
            a = atletas[0]
            conteudo = (
                f"Cartola API: {a.get('apelido','?')} | "
                f"Preço: C${a.get('preco_num', '?')} | "
                f"Média: {a.get('media_num', '?')} | "
                f"Pontos: {a.get('pontos_num', '?')} | "
                f"Variação: {a.get('variacao_num', '?')}"
            )
            return FonteResultado(
                fonte="cartola_api",
                url=url,
                status="ok",
                dados=a,
                conteudo_bruto=conteudo,
                tempo_ms=(time.time() - inicio) * 1000,
            )
    except Exception:
        pass
    return FonteResultado(fonte="cartola_api", url=url, status="vazio", dados={},
                          tempo_ms=(time.time() - inicio) * 1000)


def _raspar_espn_escalacao(nome_time: str) -> FonteResultado:
    """Busca escalacão provável na ESPN Brasil"""
    query = nome_time.replace(" ", "+")
    url = f"https://www.espn.com.br/futebol/time/escalacoes/_/nome/{query}"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="espn_escalacao", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, [".lineup__list", ".lineup__slot", ".player-column"])
    return FonteResultado(
        fonte="espn_escalacao",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


def _raspar_whoscored(nome: str) -> FonteResultado:
    """Busca rating e série histórica no WhoScored"""
    query = nome.replace(" ", "+")
    url = f"https://www.whoscored.com/Search/?t={query}"
    inicio = time.time()
    resp = _get(url)
    if not resp:
        return FonteResultado(fonte="whoscored", url=url, status="erro", dados={},
                              tempo_ms=(time.time() - inicio) * 1000)
    soup = BeautifulSoup(resp.text, "html.parser")
    conteudo = _text_from_soup(soup, [".search-item-name", ".rating", ".statistic"])
    return FonteResultado(
        fonte="whoscored",
        url=url,
        status="ok" if conteudo else "vazio",
        dados={},
        conteudo_bruto=conteudo[:600],
        tempo_ms=(time.time() - inicio) * 1000,
    )


# ---------------------------------------------------------------------------
# Interface pública principal
# ---------------------------------------------------------------------------

def pesquisar_jogador(
    nome: str,
    time: str,
    sofascore_id: Optional[int] = None,
    forcar_atualizacao: bool = False,
) -> PesquisaJogador:
    """
    Pesquisa um jogador em pelo menos 10 fontes web em paralelo.
    Retorna PesquisaJogador consolidado com contexto para o LLM.

    Args:
        nome:              Nome/apelido do jogador (ex: "Hulk")
        time:              Nome do time (ex: "Atlético-MG")
        sofascore_id:      ID no Sofascore (opcional, melhora a busca)
        forcar_atualizacao: Ignora cache e refaz todas as requisições
    """
    cache_key = hashlib.md5(f"{nome}:{time}".encode()).hexdigest()
    if not forcar_atualizacao:
        cached = _cache.get(cache_key)
        if cached:
            logger.info(f"[cache hit] {nome}")
            return cached

    logger.info(f"Iniciando pesquisa web para {nome} ({time}) em 12 fontes...")

    tarefas = [
        ("sofascore",     lambda: _raspar_sofascore_jogador(nome, sofascore_id)),
        ("ge_noticias",   lambda: _raspar_ge_noticias(f"{nome} {time}")),
        ("ge_time",       lambda: _raspar_ge_noticias(time, tipo="time")),
        ("google_news_j", lambda: _raspar_google_news_rss(nome)),
        ("google_news_t", lambda: _raspar_google_news_rss(time)),
        ("transfermarkt", lambda: _raspar_transfermarkt(nome)),
        ("flashscore",    lambda: _raspar_flashscore(nome)),
        ("cartoleiro",    lambda: _raspar_cartoleiro(nome)),
        ("redanao",       lambda: _raspar_redanao(nome)),
        ("terrao",        lambda: _raspar_terrao(nome)),
        ("cartola_api",   lambda: _raspar_cartola_api_oficial(nome)),
        ("whoscored",     lambda: _raspar_whoscored(nome)),
    ]

    resultados: List[FonteResultado] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(fn): nome_fonte for nome_fonte, fn in tarefas}
        for futuro in as_completed(futuros, timeout=30):
            try:
                resultados.append(futuro.result())
            except Exception as exc:
                nome_fonte = futuros[futuro]
                resultados.append(FonteResultado(
                    fonte=nome_fonte, url="", status="erro",
                    dados={}, erro=str(exc)
                ))

    sucesso = sum(1 for r in resultados if r.status == "ok")
    pesquisa = PesquisaJogador(
        nome=nome,
        time=time,
        fontes_consultadas=len(resultados),
        fontes_sucesso=sucesso,
        resultados=resultados,
        gerado_em=datetime.now().isoformat(),
    )

    # Consolida dados estruturados da API oficial do Cartola (mais confiável)
    cartola_res = next((r for r in resultados if r.fonte == "cartola_api" and r.status == "ok"), None)
    if cartola_res:
        pesquisa.dados_consolidados["cartola"] = cartola_res.dados

    sofa_res = next((r for r in resultados if r.fonte == "sofascore" and r.status == "ok"), None)
    if sofa_res:
        pesquisa.dados_consolidados["sofascore"] = sofa_res.dados

    _cache.set(cache_key, pesquisa)
    logger.info(
        f"Pesquisa concluída: {nome} | {sucesso}/{len(resultados)} fontes com sucesso"
    )
    return pesquisa


def pesquisar_time(
    nome_time: str,
    forcar_atualizacao: bool = False,
) -> PesquisaTime:
    """
    Pesquisa um time em 10 fontes web em paralelo.
    """
    cache_key = hashlib.md5(f"time:{nome_time}".encode()).hexdigest()
    if not forcar_atualizacao:
        cached = _cache.get(cache_key)
        if cached:
            return cached

    logger.info(f"Pesquisando time {nome_time} em 10 fontes...")

    tarefas = [
        ("ge_noticias",        lambda: _raspar_ge_noticias(nome_time, tipo="time")),
        ("ge_jogadores",       lambda: _raspar_ge_noticias(f"{nome_time} escalacao desfalques")),
        ("google_news_time",   lambda: _raspar_google_news_rss(nome_time)),
        ("google_news_lesoes", lambda: _raspar_google_news_rss(f"{nome_time} desfalque lesao")),
        ("flashscore_time",    lambda: _raspar_flashscore(nome_time)),
        ("transfermarkt_time", lambda: _raspar_transfermarkt(nome_time)),
        ("espn_escalacao",     lambda: _raspar_espn_escalacao(nome_time)),
        ("cartoleiro_time",    lambda: _raspar_cartoleiro(nome_time)),
        ("redanao_time",       lambda: _raspar_redanao(nome_time)),
        ("whoscored_time",     lambda: _raspar_whoscored(nome_time)),
    ]

    resultados: List[FonteResultado] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(fn): n for n, fn in tarefas}
        for futuro in as_completed(futuros, timeout=30):
            try:
                resultados.append(futuro.result())
            except Exception as exc:
                resultados.append(FonteResultado(
                    fonte=futuros[futuro], url="", status="erro", dados={}, erro=str(exc)
                ))

    sucesso = sum(1 for r in resultados if r.status == "ok")
    pesquisa = PesquisaTime(
        nome=nome_time,
        fontes_consultadas=len(resultados),
        fontes_sucesso=sucesso,
        resultados=resultados,
        gerado_em=datetime.now().isoformat(),
    )
    _cache.set(cache_key, pesquisa)
    return pesquisa


def pesquisar_rodada_completa(
    jogadores: List[Dict[str, Any]],
    times: List[str],
) -> Dict[str, Any]:
    """
    Pesquisa todos os jogadores + times de uma rodada em paralelo.

    Args:
        jogadores: lista de dicts com ao menos {"nome": str, "time": str}
        times:     lista de nomes de times envolvidos na rodada

    Returns:
        {
            "jogadores": {nome: PesquisaJogador},
            "times":     {nome: PesquisaTime},
            "contexto_completo": str  <- pronto para injetar no prompt do LLM
        }
    """
    resultado: Dict[str, Any] = {"jogadores": {}, "times": {}}

    # Pesquisa paralela de jogadores
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros_j = {
            executor.submit(pesquisar_jogador, j["nome"], j.get("time", "")): j["nome"]
            for j in jogadores
        }
        for futuro in as_completed(futuros_j, timeout=120):
            nome = futuros_j[futuro]
            try:
                resultado["jogadores"][nome] = futuro.result()
            except Exception as exc:
                logger.warning(f"Falha na pesquisa de {nome}: {exc}")

    # Pesquisa paralela de times
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros_t = {executor.submit(pesquisar_time, t): t for t in times}
        for futuro in as_completed(futuros_t, timeout=60):
            time_nome = futuros_t[futuro]
            try:
                resultado["times"][time_nome] = futuro.result()
            except Exception as exc:
                logger.warning(f"Falha na pesquisa de {time_nome}: {exc}")

    # Monta contexto completo para o LLM
    blocos = []
    for nome, pj in resultado["jogadores"].items():
        blocos.append(pj.to_context_string())
    for nome, pt in resultado["times"].items():
        blocos.append(pt.to_context_string())

    resultado["contexto_completo"] = "\n\n".join(blocos)
    return resultado


# ---------------------------------------------------------------------------
# CLI de teste
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    nome = sys.argv[1] if len(sys.argv) > 1 else "Hulk"
    time_nome = sys.argv[2] if len(sys.argv) > 2 else "Atlético-MG"

    print(f"\nPesquisando {nome} ({time_nome}) em 12 fontes...\n")
    pesquisa = pesquisar_jogador(nome, time_nome)
    print(f"Fontes: {pesquisa.fontes_consultadas} | Sucesso: {pesquisa.fontes_sucesso}")
    print("\n" + pesquisa.to_context_string())
