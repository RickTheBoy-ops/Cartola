from prometheus_client import Counter, Histogram

# Histograms (track values over time like latency)
PREDICTION_LATENCY = Histogram(
    'cartola_prediction_latency_seconds', 
    'Tempo gasto gerando uma predição em segundos',
    ['model_type']
)

OPTIMIZATION_LATENCY = Histogram(
    'cartola_optimization_latency_seconds',
    'Tempo gasto pela factory de otimização em segundos',
    ['strategy']
)

# Counters (track total occurrences)
API_CALLS_TOTAL = Counter(
    'cartola_api_calls_total', 
    'Total de chamadas à API da globo',
    ['endpoint', 'status_code']
)

CACHE_HITS = Counter(
    'cartola_cache_hits_total',
    'Total de hits de operações cacheadas'
)

CACHE_MISSES = Counter(
    'cartola_cache_misses_total',
    'Total de operações não cacheadas passadas direto para API'
)
