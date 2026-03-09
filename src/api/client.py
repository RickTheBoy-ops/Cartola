import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os
import yaml
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

from src.utils.cache import api_cache, disk_cache

load_dotenv()

logger = logging.getLogger(__name__)


class CartolaAPIClient:
    """
    Cliente robusto para API do Cartola FC com:
    - Retry automático em caso de falha
    - Rate limiting para evitar bloqueio
    - Cache inteligente (memória + disco)
    - Logging detalhado
    """

    # TTLs de cache por tipo de dado
    CACHE_TTLS = {
        'clubes': 86400,          # 24h - mudam raramente
        'mercado_status': 120,    # 2min - muda com frequência
        'atletas_mercado': 300,   # 5min - preços mudam
        'partidas': 600,          # 10min
        'pontuados': 60,          # 1min - dados em tempo real
    }

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        config_path: str = "config.yaml"
    ):
        # Carregar configurações
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        api_config = self.config.get('api', {})
        self.base_url = api_config.get('base_url', "https://api.cartolafc.globo.com")
        self.timeout = api_config.get('timeout', 30)
        self.max_retries = api_config.get('max_retries', 5)
        self.rate_limit_delay = api_config.get('rate_limit_delay', 1.0)

        self.session = self._create_session(self.max_retries)
        self.glb_token = os.getenv('GLB_TOKEN')
        self.last_request_time = 0

        # Se não tiver token mas tiver credenciais, autentica
        email = email or os.getenv('CARTOLA_EMAIL')
        password = password or os.getenv('CARTOLA_PASSWORD')

        if not self.glb_token and email and password:
            self.authenticate(email, password)

    def _create_session(self, max_retries: int) -> requests.Session:
        """Cria sessão com retry automático"""
        session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        return session

    def _rate_limit(self):
        """Implementa rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        authenticated: bool = False,
        cache_key: Optional[str] = None,
        cache_ttl: int = 0
    ) -> Dict[Any, Any]:
        """Realiza requisição com tratamento de erros e cache"""

        # Checar cache primeiro
        if cache_key and cache_ttl > 0:
            cached = api_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached

        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {}

        if authenticated and self.glb_token:
            headers['X-GLB-Token'] = self.glb_token

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            response_size = len(response.content)
            logger.info(f"✅ {method} {endpoint} - Status: {response.status_code} ({response_size} bytes)")

            # Salvar no cache
            if cache_key and cache_ttl > 0:
                api_cache.set(cache_key, result, cache_ttl)

            return result

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 'N/A'
            logger.error(f"❌ HTTP Error: {status} - {endpoint}")
            if status == 404:
                raise CartolaAPIError(f"Recurso não encontrado: {endpoint}")
            elif status == 401:
                raise CartolaAPIError("Não autenticado. Verifique suas credenciais.")
            else:
                raise CartolaAPIError(f"Erro HTTP {status}: {str(e)}")

        except requests.exceptions.Timeout:
            logger.error(f"⏳ Timeout na requisição: {endpoint}")
            raise CartolaAPIError(f"Timeout ao acessar {endpoint}")

        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Erro de conexão: {endpoint}")
            raise CartolaAPIError("Erro de conexão com a API do Cartola FC")

        except Exception as e:
            logger.error(f"💥 Erro inesperado: {str(e)}")
            raise CartolaAPIError(f"Erro inesperado: {str(e)}")

    def authenticate(self, email: str, password: str):
        """Autentica na API e obtém token"""
        endpoint = "/auth/dologin"
        data = {
            "payload": {
                "email": email,
                "password": password,
                "serviceId": 4728
            }
        }

        try:
            response = self._request("POST", endpoint, data=data)
            self.glb_token = response.get('glb_id')
            logger.info("🔑 Autenticação realizada com sucesso")
        except Exception as e:
            logger.error(f"🔒 Falha na autenticação: {str(e)}")
            raise

    # ========== ENDPOINTS PÚBLICOS (com cache) ==========

    def get_mercado_status(self) -> Dict:
        """Obtém status do mercado (aberto/fechado) e rodada atual"""
        return self._request(
            "GET", "/mercado/status",
            cache_key="mercado_status",
            cache_ttl=self.CACHE_TTLS['mercado_status']
        )

    def get_atletas_mercado(self) -> Dict:
        """Obtém todos atletas disponíveis no mercado com preços e estatísticas"""
        return self._request(
            "GET", "/atletas/mercado",
            cache_key="atletas_mercado",
            cache_ttl=self.CACHE_TTLS['atletas_mercado']
        )

    def get_atletas_pontuados(self) -> Dict:
        """Obtém pontuação parcial dos atletas (mercado fechado)"""
        return self._request(
            "GET", "/atletas/pontuados",
            cache_key="atletas_pontuados",
            cache_ttl=self.CACHE_TTLS['pontuados']
        )

    def get_partidas(self, rodada: Optional[int] = None) -> Dict:
        """Obtém informações das partidas"""
        endpoint = f"/partidas/{rodada}" if rodada else "/partidas"
        return self._request(
            "GET", endpoint,
            cache_key=f"partidas_{rodada}",
            cache_ttl=self.CACHE_TTLS['partidas']
        )

    def get_clubes(self) -> Dict:
        """Obtém informações de todos os clubes (cache longo)"""
        # Tenta cache de disco primeiro (24h)
        cached = disk_cache.get("clubes", ttl_seconds=self.CACHE_TTLS['clubes'])
        if cached:
            logger.debug("Cache de disco HIT: clubes")
            return cached

        result = self._request("GET", "/clubes")
        disk_cache.set("clubes", result, self.CACHE_TTLS['clubes'])
        return result

    def get_pos_rodada_destaques(self) -> Dict:
        """Obtém destaques da rodada (mito, média de pontos, etc)"""
        return self._request("GET", "/pos-rodada/destaques")

    # ========== ENDPOINTS AUTENTICADOS ==========

    def get_time_logado(self) -> Dict:
        """Obtém time do usuário autenticado"""
        return self._request("GET", "/auth/time", authenticated=True)

    def get_liga(self, slug: str, page: int = 1, order_by: str = "campeonato") -> Dict:
        """Obtém informações de uma liga"""
        endpoint = f"/auth/liga/{slug}"
        params = {"page": page, "orderBy": order_by}
        return self._request("GET", endpoint, params=params, authenticated=True)

    def get_pontuacao_atleta(self, atleta_id: int) -> Dict:
        """Obtém histórico de pontuação de um atleta"""
        endpoint = f"/auth/mercado/atleta/{atleta_id}/pontuacao"
        return self._request("GET", endpoint, authenticated=True)

    def salvar_time(self, time_data: Dict) -> Dict:
        """Salva escalação do time"""
        return self._request("POST", "/time/salvar", data=time_data, authenticated=True)


class CartolaAPIError(Exception):
    """Exceção customizada para erros da API"""
    pass
