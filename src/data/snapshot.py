"""
Camada de coleta e cache local — NUNCA bate na API mais de uma vez por rodada.

Fluxo:
  1. Verifica se já existe snapshot local para rodada atual (dados_rodada_X.json)
  2. Se existir e não estiver expirado, carrega do disco.
  3. Caso contrário, faz UMA única chamada à API e salva tudo em disco.

Assim o algoritmo de escalação lê SEMPRE de arquivo local.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

STATUS_PROVAVEL  = 7   # Único status aceito para escalação
SNAPSHOT_DIR     = Path("data/snapshots")
SNAPSHOT_TTL_MIN = 30  # tempo de vida do cache em minutos


class RodadaSnapshot:
    """
    Garante uma única coleta por rodada, com cache em disco.

    Uso:
        snap = RodadaSnapshot(api_client)
        dados = snap.obter()   # sempre retorna do disco se possível
        atletas_df = snap.atletas_provavel()  # já filtrado por status_id=7
    """

    def __init__(self, api_client=None, snapshot_dir: Path = SNAPSHOT_DIR):
        self.api          = api_client
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._dados: Optional[Dict] = None

    # ----------------------------------------------------------
    # PÚBLICO
    # ----------------------------------------------------------

    def obter(self, forcar_reload: bool = False) -> Dict:
        """
        Retorna dados da rodada.
        - Usa cache em disco se válido (< SNAPSHOT_TTL_MIN minutos).
        - Bate na API apenas uma vez quando necessário.
        """
        if self._dados and not forcar_reload:
            return self._dados

        rodada = self._rodada_atual()
        path   = self._caminho(rodada)

        if not forcar_reload and self._cache_valido(path):
            logger.info(f"💾 Carregando snapshot local: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                self._dados = json.load(f)
            return self._dados

        logger.info("📡 Coletando dados frescos da API Cartola (uma única vez)...")
        self._dados = self._coletar_api(rodada)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._dados, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Snapshot salvo: {path}")
        return self._dados

    def atletas_provaveis(self) -> list:
        """
        Retorna APENAS atletas com status_id == 7 (Provável).
        Filtra na borda — jogadores duvidosos nunca chegam ao otimizador.
        """
        dados   = self.obter()
        atletas = dados.get('atletas', [])
        if isinstance(atletas, dict):
            atletas = list(atletas.values())

        total   = len(atletas)
        filtrado = [a for a in atletas if int(a.get('status_id', 0)) == STATUS_PROVAVEL]
        removidos = total - len(filtrado)

        logger.info(
            f"🧹 Filtro status_id={STATUS_PROVAVEL}: "
            f"{len(filtrado)} prováveis | {removidos} ignorados "
            f"(dúvida/contundido/suspenso)"
        )
        return filtrado

    def partidas(self) -> list:
        return self.obter().get('partidas', [])

    def rodada_atual(self) -> int:
        return int(self.obter().get('rodada_atual', 0))

    def status_mercado(self) -> dict:
        return self.obter().get('mercado_status', {})

    # ----------------------------------------------------------
    # PRIVADO
    # ----------------------------------------------------------

    def _rodada_atual(self) -> int:
        """Descobre rodada atual via API (chamada leve de mercado/status)."""
        try:
            if self.api:
                status = self.api.get_mercado_status()
                return int(status.get('rodada_atual', 1))
            r = requests.get(
                "https://api.cartola.globo.com/mercado/status",
                headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
            )
            return int(r.json().get('rodada_atual', 1))
        except Exception as e:
            logger.warning(f"⚠️  Não foi possível obter rodada atual: {e}. Usando rodada 1.")
            return 1

    def _caminho(self, rodada: int) -> Path:
        return self.snapshot_dir / f"dados_rodada_{rodada}.json"

    def _cache_valido(self, path: Path) -> bool:
        if not path.exists():
            return False
        idade = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return idade < timedelta(minutes=SNAPSHOT_TTL_MIN)

    def _coletar_api(self, rodada: int) -> Dict:
        """Faz UMA coleta completa e estruturada da API."""
        dados: Dict = {'rodada_atual': rodada, 'coletado_em': datetime.now().isoformat()}

        # Mercado status
        try:
            if self.api:
                dados['mercado_status'] = self.api.get_mercado_status()
            else:
                r = requests.get(
                    "https://api.cartola.globo.com/mercado/status",
                    headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
                )
                dados['mercado_status'] = r.json()
        except Exception as e:
            logger.warning(f"⚠️  mercado/status falhou: {e}")
            dados['mercado_status'] = {}

        # Atletas do mercado (dados brutos completos)
        try:
            if self.api:
                raw = self.api.get_atletas_mercado()
            else:
                r = requests.get(
                    "https://api.cartola.globo.com/atletas/mercado",
                    headers={'User-Agent': 'Mozilla/5.0'}, timeout=30
                )
                raw = r.json()
            atletas = raw.get('atletas', raw)
            dados['atletas'] = list(atletas.values()) if isinstance(atletas, dict) else atletas
            logger.info(f"📥 {len(dados['atletas'])} atletas brutos recebidos")
        except Exception as e:
            logger.error(f"❌ atletas/mercado falhou: {e}")
            dados['atletas'] = []

        # Partidas da rodada
        try:
            if self.api:
                partidas_raw = self.api.get_partidas(rodada)
            else:
                r = requests.get(
                    f"https://api.cartola.globo.com/partidas/{rodada}",
                    headers={'User-Agent': 'Mozilla/5.0'}, timeout=15
                )
                partidas_raw = r.json()
            dados['partidas'] = partidas_raw.get('partidas', [])
            logger.info(f"⚽ {len(dados['partidas'])} partidas coletadas")
        except Exception as e:
            logger.warning(f"⚠️  partidas falhou: {e}")
            dados['partidas'] = []

        # Clubes
        try:
            r = requests.get(
                "https://api.cartola.globo.com/clubes",
                headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
            )
            dados['clubes'] = r.json()
        except Exception as e:
            logger.warning(f"⚠️  clubes falhou: {e}")
            dados['clubes'] = {}

        return dados
