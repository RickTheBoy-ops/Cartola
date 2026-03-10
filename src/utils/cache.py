"""Sistema de cache para reduzir chamadas à API e melhorar performance."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import pickle

logger = logging.getLogger(__name__)


class CacheManager:
    """Gerenciador de cache em disco com TTL (Time To Live)."""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 300):
        """
        Args:
            cache_dir: Diretório para armazenar arquivos de cache
            default_ttl: Tempo de vida padrão em segundos (5 minutos)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
    
    def _get_cache_path(self, key: str) -> Path:
        """Gera caminho do arquivo de cache baseado na chave."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def _is_expired(self, cache_data: dict) -> bool:
        """Verifica se o cache expirou."""
        if 'expires_at' not in cache_data:
            return True
        
        expires_at = datetime.fromisoformat(cache_data['expires_at'])
        return datetime.now() > expires_at
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache.
        
        Args:
            key: Chave única do cache
            
        Returns:
            Valor armazenado ou None se não encontrado/expirado
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {key}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if self._is_expired(cache_data):
                logger.debug(f"Cache expired: {key}")
                cache_path.unlink()  # Remove cache expirado
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit: {key}")
            return cache_data['value']
            
        except Exception as e:
            logger.error(f"Erro ao ler cache {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Armazena valor no cache.
        
        Args:
            key: Chave única do cache
            value: Valor a ser armazenado
            ttl: Tempo de vida em segundos (usa default_ttl se None)
            
        Returns:
            True se sucesso, False se erro
        """
        cache_path = self._get_cache_path(key)
        ttl = ttl or self.default_ttl
        
        cache_data = {
            'value': value,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            'ttl': ttl
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar cache {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Remove item do cache."""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Cache deleted: {key}")
                return True
            except Exception as e:
                logger.error(f"Erro ao deletar cache {key}: {e}")
                return False
        
        return False
    
    def clear(self) -> int:
        """Remove todos os arquivos de cache.
        
        Returns:
            Número de arquivos removidos
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Erro ao deletar {cache_file}: {e}")
        
        logger.info(f"Cache cleared: {count} arquivos removidos")
        return count
    
    def get_stats(self) -> dict:
        """Retorna estatísticas de uso do cache."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': round(hit_rate, 2),
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calcula tamanho total do cache em MB."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
        return round(total_size / (1024 * 1024), 2)


# Instância global do cache
_cache_manager = CacheManager()


def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator para cachear resultados de funções.
    
    Args:
        ttl: Tempo de vida do cache em segundos
        key_prefix: Prefixo para a chave do cache
        
    Example:
        @cached(ttl=600, key_prefix="atletas")
        def get_atletas(rodada):
            return api.fetch_atletas(rodada)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gerar chave única baseada em função e argumentos
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            cache_key = ":".join(filter(None, key_parts))
            
            # Tentar obter do cache
            cached_value = _cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Executar função e cachear resultado
            result = func(*args, **kwargs)
            _cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


def clear_cache():
    """Limpa todo o cache."""
    return _cache_manager.clear()


def get_cache_stats() -> dict:
    """Retorna estatísticas do cache."""
    return _cache_manager.get_stats()
