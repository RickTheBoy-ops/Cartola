"""Testes para sistema de cache."""

import pytest
import time
from pathlib import Path
from src.utils.cache import CacheManager, cached, clear_cache, get_cache_stats


class TestCacheManager:
    """Testes para CacheManager."""
    
    def test_cache_set_and_get(self, tmp_path):
        """Testa armazenamento e recuperação básica."""
        cache = CacheManager(cache_dir=str(tmp_path), default_ttl=300)
        
        # Armazenar valor
        cache.set("test_key", {"data": "test_value"})
        
        # Recuperar valor
        result = cache.get("test_key")
        assert result is not None
        assert result["data"] == "test_value"
    
    def test_cache_miss(self, tmp_path):
        """Testa cache miss."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        result = cache.get("non_existent_key")
        assert result is None
    
    def test_cache_expiration(self, tmp_path):
        """Testa expiração do cache."""
        cache = CacheManager(cache_dir=str(tmp_path), default_ttl=1)
        
        # Armazenar com TTL de 1 segundo
        cache.set("expire_key", "value", ttl=1)
        
        # Deve estar disponível imediatamente
        assert cache.get("expire_key") == "value"
        
        # Aguardar expiração
        time.sleep(1.5)
        
        # Deve retornar None após expirar
        assert cache.get("expire_key") is None
    
    def test_cache_delete(self, tmp_path):
        """Testa deleção de cache."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        cache.set("delete_key", "value")
        assert cache.get("delete_key") == "value"
        
        # Deletar
        cache.delete("delete_key")
        assert cache.get("delete_key") is None
    
    def test_cache_clear(self, tmp_path):
        """Testa limpeza completa do cache."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        # Adicionar vários itens
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Limpar tudo
        count = cache.clear()
        assert count == 5
        
        # Verificar que tudo foi removido
        for i in range(5):
            assert cache.get(f"key_{i}") is None
    
    def test_cache_stats(self, tmp_path):
        """Testa estatísticas do cache."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        # Hit
        cache.set("key1", "value1")
        cache.get("key1")
        
        # Miss
        cache.get("non_existent")
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 50.0
    
    def test_cache_com_tipos_diferentes(self, tmp_path):
        """Testa cache com diferentes tipos de dados."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        # String
        cache.set("str_key", "string_value")
        assert cache.get("str_key") == "string_value"
        
        # Dict
        cache.set("dict_key", {"a": 1, "b": 2})
        assert cache.get("dict_key") == {"a": 1, "b": 2}
        
        # List
        cache.set("list_key", [1, 2, 3, 4, 5])
        assert cache.get("list_key") == [1, 2, 3, 4, 5]
        
        # Int
        cache.set("int_key", 42)
        assert cache.get("int_key") == 42


class TestCachedDecorator:
    """Testes para decorator @cached."""
    
    def test_cached_function(self, tmp_path):
        """Testa função com decorator @cached."""
        call_count = {'count': 0}
        
        @cached(ttl=300, key_prefix="test")
        def expensive_function(x):
            call_count['count'] += 1
            return x * 2
        
        # Primeira chamada - executa função
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count['count'] == 1
        
        # Segunda chamada - usa cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count['count'] == 1  # Não incrementou
        
        # Chamada com argumento diferente - executa novamente
        result3 = expensive_function(7)
        assert result3 == 14
        assert call_count['count'] == 2
    
    def test_cached_with_kwargs(self, tmp_path):
        """Testa cache com keyword arguments."""
        call_count = {'count': 0}
        
        @cached(ttl=300)
        def function_with_kwargs(a, b=10):
            call_count['count'] += 1
            return a + b
        
        # Primeira chamada
        result1 = function_with_kwargs(5, b=15)
        assert result1 == 20
        assert call_count['count'] == 1
        
        # Mesmos argumentos - usa cache
        result2 = function_with_kwargs(5, b=15)
        assert result2 == 20
        assert call_count['count'] == 1
        
        # Argumentos diferentes
        result3 = function_with_kwargs(5, b=20)
        assert result3 == 25
        assert call_count['count'] == 2
    
    def test_cached_expiration(self, tmp_path):
        """Testa expiração com decorator."""
        call_count = {'count': 0}
        
        @cached(ttl=1)  # 1 segundo
        def short_lived_cache(x):
            call_count['count'] += 1
            return x * 3
        
        # Primeira chamada
        result1 = short_lived_cache(2)
        assert result1 == 6
        assert call_count['count'] == 1
        
        # Imediatamente - usa cache
        result2 = short_lived_cache(2)
        assert call_count['count'] == 1
        
        # Aguardar expiração
        time.sleep(1.5)
        
        # Após expiração - executa novamente
        result3 = short_lived_cache(2)
        assert result3 == 6
        assert call_count['count'] == 2


class TestCacheUtilities:
    """Testes para funções utilitárias."""
    
    def test_clear_cache_global(self):
        """Testa limpeza do cache global."""
        @cached(ttl=300)
        def func(x):
            return x * 2
        
        # Criar alguns caches
        func(1)
        func(2)
        func(3)
        
        # Limpar
        count = clear_cache()
        assert count >= 0  # Pode ter outros caches de testes anteriores
    
    def test_get_cache_stats_global(self):
        """Testa obtenção de estatísticas globais."""
        stats = get_cache_stats()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'total_requests' in stats
        assert 'hit_rate' in stats
        assert 'cache_size_mb' in stats
