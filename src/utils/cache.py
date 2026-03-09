"""
Sistema de Cache TTL para o Cartola FC Optimizer
- Cache em memória com expiração configurável
- Cache de disco para dados persistentes (clubes, posições)
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """Cache em memória com TTL (Time-To-Live)"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retorna valor do cache se existir e não expirado"""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if time.time() > entry['expires_at']:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry['value']

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Armazena valor no cache com TTL em segundos"""
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl_seconds,
            'created_at': time.time()
        }

    def invalidate(self, key: str):
        """Remove chave do cache"""
        self._cache.pop(key, None)

    def clear(self):
        """Limpa todo o cache"""
        self._cache.clear()

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%",
            'entries': len(self._cache)
        }


class DiskCache:
    """Cache em disco para dados que mudam raramente"""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, ttl_seconds: int = 86400) -> Optional[Any]:
        """Retorna dados do disco se existirem e não expiraram"""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if time.time() > data.get('expires_at', 0):
                path.unlink(missing_ok=True)
                return None
            return data['value']
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 86400):
        """Salva dados no disco com TTL (padrão: 24h)"""
        path = self._key_to_path(key)
        data = {
            'value': value,
            'expires_at': time.time() + ttl_seconds,
            'created_at': time.time(),
            'key': key
        }
        path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding='utf-8')
        logger.debug(f"Cache de disco salvo: {key}")

    def clear(self):
        """Limpa todo o cache de disco"""
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)


# Instâncias globais reutilizáveis
api_cache = MemoryCache()
disk_cache = DiskCache()
