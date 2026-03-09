import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path
import yaml

from src.utils.validators import validar_atleta, filtrar_atletas_validos

logger = logging.getLogger(__name__)


class CartolaDataCollector:
    """
    Coletor de dados com persistência em banco de dados SQLite.
    - Parsing robusto da API (suporta dict e list)
    - Queries parametrizadas (sem SQL injection)
    - Validação de dados antes de inserir
    """

    def __init__(self, api_client, config_path: str = "config.yaml"):
        self.api = api_client

        # Carregar configurações
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        db_path = self.config.get('database', {}).get('path', "data/cartola.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Inicializa schema do banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela de Atletas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atletas (
                atleta_id INTEGER PRIMARY KEY,
                nome TEXT,
                apelido TEXT,
                clube_id INTEGER,
                posicao_id INTEGER,
                status_id INTEGER DEFAULT 7,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Adicionar coluna status_id caso a tabela já exista sem ela originariamente
        try:
            cursor.execute("ALTER TABLE atletas ADD COLUMN status_id INTEGER DEFAULT 7")
        except sqlite3.OperationalError:
            pass # Coluna já existe

        # Tabela de Pontuações por Rodada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pontuacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                atleta_id INTEGER,
                rodada INTEGER,
                pontos REAL,
                preco REAL,
                variacao REAL,
                media REAL,
                jogos INTEGER,
                
                -- Scouts positivos
                G INTEGER DEFAULT 0,   -- Gol
                A INTEGER DEFAULT 0,   -- Assistência
                SG INTEGER DEFAULT 0,  -- Sem sofrer gol
                FT INTEGER DEFAULT 0,  -- Finalização na trave
                FD INTEGER DEFAULT 0,  -- Finalização defendida
                FF INTEGER DEFAULT 0,  -- Finalização fora
                FS INTEGER DEFAULT 0,  -- Falta sofrida
                PE INTEGER DEFAULT 0,  -- Pênalti ganho
                I INTEGER DEFAULT 0,   -- Impedimento provocado
                PP INTEGER DEFAULT 0,  -- Pênalti perdido (negativo)
                DS INTEGER DEFAULT 0,  -- Desarme
                DE INTEGER DEFAULT 0,  -- Defesa
                DP INTEGER DEFAULT 0,  -- Defesa de pênalti
                GS INTEGER DEFAULT 0,  -- Gol sofrido (negativo)
                FC INTEGER DEFAULT 0,  -- Falta cometida
                GC INTEGER DEFAULT 0,  -- Gol contra (negativo)
                CA INTEGER DEFAULT 0,  -- Cartão amarelo (negativo)
                CV INTEGER DEFAULT 0,  -- Cartão vermelho (negativo)
                
                minutos_jogados INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (atleta_id) REFERENCES atletas(atleta_id),
                UNIQUE(atleta_id, rodada)
            )
        """)

        # Tabela de Partidas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partidas (
                partida_id INTEGER PRIMARY KEY,
                rodada INTEGER,
                clube_casa_id INTEGER,
                clube_visitante_id INTEGER,
                placar_oficial_mandante INTEGER,
                placar_oficial_visitante INTEGER,
                aproveitamento_mandante REAL,
                aproveitamento_visitante REAL,
                valid BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(partida_id, rodada)
            )
        """)

        # Tabela de Mercado Status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mercado_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rodada_atual INTEGER,
                status_id INTEGER,
                status_nome TEXT,
                times_escalados INTEGER,
                fechamento DATETIME,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Índices para performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pontuacoes_atleta ON pontuacoes(atleta_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pontuacoes_rodada ON pontuacoes(rodada)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_rodada ON partidas(rodada)")

        conn.commit()
        conn.close()

        logger.info(f"💾 Database inicializado: {self.db_path}")

    def _parse_atletas(self, data: Dict) -> List[Dict]:
        """
        Parse robusto dos atletas da API.
        A API pode retornar:
          - Lista: [{"atleta_id": 1, ...}, ...]
          - Dicionário: {"123": {"atleta_id": 1, ...}, ...}
        """
        atletas_raw = data.get('atletas', data)

        if isinstance(atletas_raw, dict):
            # API retorna dict com IDs como chaves
            return list(atletas_raw.values())
        elif isinstance(atletas_raw, list):
            return atletas_raw
        else:
            logger.warning(f"⚠️ Formato inesperado de atletas: {type(atletas_raw)}")
            return []

    def collect_mercado_status(self) -> Dict:
        """Coleta e armazena status do mercado"""
        try:
            status = self.api.get_mercado_status()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO mercado_status (rodada_atual, status_id, status_nome, times_escalados, fechamento)
                VALUES (?, ?, ?, ?, ?)
            """, (
                status.get('rodada_atual'),
                status.get('status_mercado'),
                status.get('nome_status'),
                status.get('times_escalados'),
                status.get('fechamento', {}).get('timestamp') if isinstance(status.get('fechamento'), dict) else status.get('fechamento')
            ))

            conn.commit()
            conn.close()

            logger.info(f"📊 Mercado status coletado - Rodada: {status.get('rodada_atual')}")
            return status

        except Exception as e:
            logger.error(f"❌ Erro ao coletar mercado status: {str(e)}")
            raise

    def collect_atletas_mercado(self, rodada: int) -> pd.DataFrame:
        """Coleta atletas do mercado com parsing robusto e validação"""
        try:
            data = self.api.get_atletas_mercado()

            # Parse robusto (suporta dict e list)
            atletas_raw = self._parse_atletas(data)
            logger.info(f"📥 API retornou {len(atletas_raw)} atletas brutos")

            atletas_list = []
            invalidos = 0
            for atleta in atletas_raw:
                # Validação básica
                if not validar_atleta(atleta):
                    invalidos += 1
                    continue

                atletas_list.append({
                    'atleta_id': atleta['atleta_id'],
                    'nome': atleta.get('nome', ''),
                    'apelido': atleta.get('apelido', ''),
                    'clube_id': atleta.get('clube_id', 0),
                    'posicao_id': atleta.get('posicao_id', 0),
                    'status_id': atleta.get('status_id', 7),
                    'rodada': rodada,
                    'pontos': atleta.get('pontos_num', 0),
                    'preco': atleta.get('preco_num', 0),
                    'variacao': atleta.get('variacao_num', 0),
                    'media': atleta.get('media_num', 0),
                    'jogos': atleta.get('jogos_num', 0),
                    'minutos_jogados': atleta.get('minutos_jogados', 0)
                })

            if invalidos > 0:
                logger.info(f"⚠️ {invalidos} atletas ignorados por dados inválidos")

            df = pd.DataFrame(atletas_list)

            if len(df) == 0:
                logger.warning("⚠️ Nenhum atleta válido encontrado!")
                return df

            # Salvar no banco com queries parametrizadas
            conn = sqlite3.connect(self.db_path)

            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO atletas (atleta_id, nome, apelido, clube_id, posicao_id, status_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (row['atleta_id'], row['nome'], row['apelido'],
                      row['clube_id'], row['posicao_id'], row.get('status_id', 7)))

            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO pontuacoes 
                        (atleta_id, rodada, pontos, preco, variacao, media, jogos, minutos_jogados)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['atleta_id'], row['rodada'], row['pontos'],
                        row['preco'], row['variacao'], row['media'],
                        row['jogos'], row['minutos_jogados']
                    ))
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao inserir atleta {row['atleta_id']}: {str(e)}")

            conn.commit()
            conn.close()

            logger.info(f"✅ Coletados {len(df)} atletas válidos da rodada {rodada}")
            return df

        except Exception as e:
            logger.error(f"❌ Erro ao coletar atletas: {str(e)}")
            raise

    def collect_atletas_pontuados(self, rodada: int) -> pd.DataFrame:
        """Coleta pontuação parcial dos atletas"""
        try:
            data = self.api.get_atletas_pontuados()

            pontuados = data.get('atletas', data)

            # Suporte para dict e list
            if isinstance(pontuados, list):
                items = [(str(a.get('atleta_id', '')), a) for a in pontuados]
            else:
                items = pontuados.items()

            conn = sqlite3.connect(self.db_path)

            for atleta_id, atleta_data in items:
                scout = atleta_data.get('scout', {})

                conn.execute("""
                    UPDATE pontuacoes SET
                        pontos = ?,
                        G = ?, A = ?, SG = ?, FT = ?, FD = ?, FF = ?,
                        FS = ?, PE = ?, I = ?, PP = ?, DS = ?, DE = ?,
                        DP = ?, GS = ?, FC = ?, GC = ?, CA = ?, CV = ?
                    WHERE atleta_id = ? AND rodada = ?
                """, (
                    atleta_data.get('pontos', 0),
                    scout.get('G', 0), scout.get('A', 0), scout.get('SG', 0),
                    scout.get('FT', 0), scout.get('FD', 0), scout.get('FF', 0),
                    scout.get('FS', 0), scout.get('PE', 0), scout.get('I', 0),
                    scout.get('PP', 0), scout.get('DS', 0), scout.get('DE', 0),
                    scout.get('DP', 0), scout.get('GS', 0), scout.get('FC', 0),
                    scout.get('GC', 0), scout.get('CA', 0), scout.get('CV', 0),
                    int(atleta_id), rodada
                ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Pontuações parciais atualizadas - Rodada {rodada}")

            # Retornar DataFrame com query parametrizada
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT * FROM pontuacoes WHERE rodada = ?",
                conn,
                params=(rodada,)
            )
            conn.close()
            return df

        except Exception as e:
            logger.error(f"❌ Erro ao coletar pontuações parciais: {str(e)}")
            raise

    def collect_partidas(self, rodada: int):
        """Coleta informações das partidas da rodada"""
        try:
            data = self.api.get_partidas(rodada)

            conn = sqlite3.connect(self.db_path)

            partidas = data.get('partidas', [])
            for partida in partidas:
                # Calcular aproveitamento de forma segura
                aprov_mandante = self._calcular_aproveitamento(partida.get('aproveitamento_mandante'))
                aprov_visitante = self._calcular_aproveitamento(partida.get('aproveitamento_visitante'))

                conn.execute("""
                    INSERT OR REPLACE INTO partidas
                    (partida_id, rodada, clube_casa_id, clube_visitante_id,
                     placar_oficial_mandante, placar_oficial_visitante,
                     aproveitamento_mandante, aproveitamento_visitante, valid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    partida.get('partida_id'),
                    rodada,
                    partida.get('clube_casa_id'),
                    partida.get('clube_visitante_id'),
                    partida.get('placar_oficial_mandante'),
                    partida.get('placar_oficial_visitante'),
                    aprov_mandante,
                    aprov_visitante,
                    partida.get('valid')
                ))

            conn.commit()
            conn.close()

            logger.info(f"⚽ {len(partidas)} partidas coletadas - Rodada {rodada}")

        except Exception as e:
            logger.error(f"❌ Erro ao coletar partidas: {str(e)}")
            raise

    @staticmethod
    def _calcular_aproveitamento(aproveitamento) -> float:
        """Calcula aproveitamento de forma segura, suportando lista ou float"""
        if aproveitamento is None:
            return 0.0
        if isinstance(aproveitamento, (int, float)):
            return float(aproveitamento)
        if isinstance(aproveitamento, list) and len(aproveitamento) > 0:
            pontos = sum(
                3 if r.lower() == 'v' else 1 if r.lower() == 'e' else 0
                for r in aproveitamento
            )
            return pontos / (len(aproveitamento) * 3)
        return 0.0

    def get_historico_atleta(self, atleta_id: int, ultimas_n_rodadas: Optional[int] = None) -> pd.DataFrame:
        """Obtém histórico de pontuações de um atleta (query parametrizada)"""
        conn = sqlite3.connect(self.db_path)

        if ultimas_n_rodadas:
            query = """
                SELECT * FROM pontuacoes 
                WHERE atleta_id = ?
                ORDER BY rodada DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(atleta_id, ultimas_n_rodadas))
        else:
            query = """
                SELECT * FROM pontuacoes 
                WHERE atleta_id = ?
                ORDER BY rodada DESC
            """
            df = pd.read_sql_query(query, conn, params=(atleta_id,))

        conn.close()
        return df
