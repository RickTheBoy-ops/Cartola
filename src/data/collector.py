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
    Coletor de dados com persistência em banco SQLite.
    - Parsing robusto da API (suporta dict e list)
    - Queries parametrizadas
    - Validação antes de inserir
    - Salva placar_a / placar_b para cruzamento H2H
    """

    def __init__(self, api_client, config_path: str = "config.yaml"):
        self.api = api_client
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        db_path = self.config.get('database', {}).get('path', "data/cartola.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Inicializa schema do banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Atletas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atletas (
                atleta_id INTEGER PRIMARY KEY,
                nome TEXT,
                apelido TEXT,
                clube_id INTEGER,
                posicao_id INTEGER,
                status_id INTEGER DEFAULT 7,
                minimo_para_valorizar REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for col in ["status_id INTEGER DEFAULT 7", "minimo_para_valorizar REAL DEFAULT 0.0"]:
            try:
                cursor.execute(f"ALTER TABLE atletas ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass

        # Migração estrutural para 'pontuacoes' (adicionar 'ano' e mudar UNIQUE)
        cursor.execute("PRAGMA table_info(pontuacoes)")
        pt_cols = [r[1] for r in cursor.fetchall()]
        needs_migration_pontuacoes = bool(pt_cols and "ano" not in pt_cols)
        if needs_migration_pontuacoes:
            logger.info("Executando migration na tabela pontuacoes: adicionando coluna 'ano' e atualizando UNIQUE(ano, rodada, atleta_id)")
            cursor.execute("ALTER TABLE pontuacoes RENAME TO pontuacoes_old")

        # Pontuações
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pontuacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                atleta_id INTEGER,
                rodada INTEGER,
                ano INTEGER DEFAULT 2024,
                pontos REAL,
                preco REAL,
                variacao REAL,
                media REAL,
                jogos INTEGER,
                G INTEGER DEFAULT 0,
                A INTEGER DEFAULT 0,
                SG INTEGER DEFAULT 0,
                FT INTEGER DEFAULT 0,
                FD INTEGER DEFAULT 0,
                FF INTEGER DEFAULT 0,
                FS INTEGER DEFAULT 0,
                PE INTEGER DEFAULT 0,
                I INTEGER DEFAULT 0,
                PP INTEGER DEFAULT 0,
                DS INTEGER DEFAULT 0,
                DE INTEGER DEFAULT 0,
                DP INTEGER DEFAULT 0,
                GS INTEGER DEFAULT 0,
                FC INTEGER DEFAULT 0,
                GC INTEGER DEFAULT 0,
                CA INTEGER DEFAULT 0,
                CV INTEGER DEFAULT 0,
                minutos_jogados INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (atleta_id) REFERENCES atletas(atleta_id),
                UNIQUE(ano, rodada, atleta_id)
            )
        """)
        
        if needs_migration_pontuacoes:
            cursor.execute("PRAGMA table_info(pontuacoes_old)")
            old_cols = [r[1] for r in cursor.fetchall() if r[1] != 'id']
            cols_str = ", ".join(old_cols)
            cursor.execute(f"INSERT INTO pontuacoes ({cols_str}) SELECT {cols_str} FROM pontuacoes_old")
            cursor.execute("DROP TABLE pontuacoes_old")

        # Migração estrutural para 'partidas' (adicionar 'ano' e mudar UNIQUE)
        cursor.execute("PRAGMA table_info(partidas)")
        pa_cols = [r[1] for r in cursor.fetchall()]
        needs_migration_partidas = bool(pa_cols and "ano" not in pa_cols)
        if needs_migration_partidas:
            logger.info("Executando migration na tabela partidas: adicionando coluna 'ano' e atualizando UNIQUE(ano, rodada, clube_casa_id, clube_visitante_id)")
            cursor.execute("ALTER TABLE partidas RENAME TO partidas_old")

        # Partidas — inclui placar_a / placar_b para H2H
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partidas (
                partida_id INTEGER PRIMARY KEY,
                rodada INTEGER,
                ano INTEGER DEFAULT 2024,
                clube_casa_id  INTEGER,
                clube_visitante_id INTEGER,
                clube_id_a INTEGER,
                clube_id_b INTEGER,
                placar_oficial_mandante INTEGER,
                placar_oficial_visitante INTEGER,
                placar_a INTEGER,
                placar_b INTEGER,
                aproveitamento_mandante REAL,
                aproveitamento_visitante REAL,
                valid BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ano, rodada, clube_casa_id, clube_visitante_id)
            )
        """)
        
        if needs_migration_partidas:
            cursor.execute("PRAGMA table_info(partidas_old)")
            old_cols = [r[1] for r in cursor.fetchall() if r[1] != 'partida_id']
            cols_str = ", ".join(old_cols)
            # partida_id é preservado
            cursor.execute(f"INSERT INTO partidas (partida_id, {cols_str}) SELECT partida_id, {cols_str} FROM partidas_old")
            cursor.execute("DROP TABLE partidas_old")
        # Adicionar colunas de H2H em bancos antigos
        for col in [
            "clube_id_a INTEGER",
            "clube_id_b INTEGER",
            "placar_a INTEGER",
            "placar_b INTEGER",
        ]:
            try:
                cursor.execute(f"ALTER TABLE partidas ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass

        # Mercado Status
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

        # Índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pontuacoes_atleta ON pontuacoes(atleta_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pontuacoes_rodada ON pontuacoes(rodada)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_rodada ON partidas(rodada)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_partidas_h2h ON partidas(clube_id_a, clube_id_b)")

        conn.commit()
        conn.close()
        logger.info(f"💾 Database inicializado: {self.db_path}")

    def _parse_atletas(self, data: Dict) -> List[Dict]:
        atletas_raw = data.get('atletas', data)
        if isinstance(atletas_raw, dict):
            return list(atletas_raw.values())
        elif isinstance(atletas_raw, list):
            return atletas_raw
        logger.warning(f"⚠️  Formato inesperado de atletas: {type(atletas_raw)}")
        return []

    def collect_mercado_status(self) -> Dict:
        try:
            status = self.api.get_mercado_status()
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO mercado_status (rodada_atual, status_id, status_nome, times_escalados, fechamento)
                VALUES (?, ?, ?, ?, ?)
            """, (
                status.get('rodada_atual'),
                status.get('status_mercado'),
                status.get('nome_status'),
                status.get('times_escalados'),
                status.get('fechamento', {}).get('timestamp')
                    if isinstance(status.get('fechamento'), dict)
                    else status.get('fechamento')
            ))
            conn.commit()
            conn.close()
            logger.info(f"📊 Mercado status coletado - Rodada: {status.get('rodada_atual')}")
            return status
        except Exception as e:
            logger.error(f"❌ Erro ao coletar mercado status: {e}")
            raise

    def collect_atletas_mercado(self, rodada: int) -> pd.DataFrame:
        try:
            data = self.api.get_atletas_mercado()
            atletas_raw = self._parse_atletas(data)
            logger.info(f"📥 API retornou {len(atletas_raw)} atletas brutos")

            atletas_list, invalidos = [], 0
            for atleta in atletas_raw:
                if not validar_atleta(atleta):
                    invalidos += 1
                    continue
                atletas_list.append({
                    'atleta_id':           atleta['atleta_id'],
                    'nome':                atleta.get('nome', ''),
                    'apelido':             atleta.get('apelido', ''),
                    'clube_id':            atleta.get('clube_id', 0),
                    'posicao_id':          atleta.get('posicao_id', 0),
                    'status_id':           atleta.get('status_id', 7),
                    'rodada':              rodada,
                    'pontos':              atleta.get('pontos_num', 0),
                    'preco':               atleta.get('preco_num', 0),
                    'variacao':            atleta.get('variacao_num', 0),
                    'media':               atleta.get('media_num', 0),
                    'jogos':               atleta.get('jogos_num', 0),
                    'minutos_jogados':     atleta.get('minutos_jogados', 0),
                    'minimo_para_valorizar': atleta.get('minimo_para_valorizar', 0.0),
                })

            if invalidos:
                logger.info(f"⚠️  {invalidos} atletas ignorados por dados inválidos")

            df = pd.DataFrame(atletas_list)
            if df.empty:
                logger.warning("⚠️  Nenhum atleta válido encontrado!")
                return df

            conn = sqlite3.connect(self.db_path)
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO atletas
                    (atleta_id, nome, apelido, clube_id, posicao_id, status_id, minimo_para_valorizar)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['atleta_id'], row['nome'], row['apelido'],
                    row['clube_id'], row['posicao_id'],
                    row.get('status_id', 7), row.get('minimo_para_valorizar', 0.0)
                ))
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
                except Exception as ex:
                    logger.warning(f"⚠️  Erro ao inserir atleta {row['atleta_id']}: {ex}")
            conn.commit()
            conn.close()

            logger.info(f"✅ {len(df)} atletas válidos coletados - rodada {rodada}")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao coletar atletas: {e}")
            raise

    def collect_atletas_pontuados(self, rodada: int) -> pd.DataFrame:
        try:
            data = self.api.get_atletas_pontuados()
            pontuados = data.get('atletas', data)
            items = (
                [(str(a.get('atleta_id', '')), a) for a in pontuados]
                if isinstance(pontuados, list)
                else pontuados.items()
            )
            conn = sqlite3.connect(self.db_path)
            for atleta_id, atleta_data in items:
                scout = atleta_data.get('scout', {})
                conn.execute("""
                    UPDATE pontuacoes SET
                        pontos=?, G=?, A=?, SG=?, FT=?, FD=?, FF=?,
                        FS=?, PE=?, I=?, PP=?, DS=?, DE=?,
                        DP=?, GS=?, FC=?, GC=?, CA=?, CV=?
                    WHERE atleta_id=? AND rodada=?
                """, (
                    atleta_data.get('pontos', 0),
                    scout.get('G',0), scout.get('A',0), scout.get('SG',0),
                    scout.get('FT',0), scout.get('FD',0), scout.get('FF',0),
                    scout.get('FS',0), scout.get('PE',0), scout.get('I',0),
                    scout.get('PP',0), scout.get('DS',0), scout.get('DE',0),
                    scout.get('DP',0), scout.get('GS',0), scout.get('FC',0),
                    scout.get('GC',0), scout.get('CA',0), scout.get('CV',0),
                    int(atleta_id), rodada
                ))
            conn.commit()
            conn.close()
            logger.info(f"✅ Pontuações parciais atualizadas - Rodada {rodada}")

            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT * FROM pontuacoes WHERE rodada=?", conn, params=(rodada,)
            )
            conn.close()
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao coletar pontuações parciais: {e}")
            raise

    def collect_partidas(self, rodada: int):
        """
        Coleta partidas e salva placar_a/placar_b + clube_id_a/clube_id_b
        para que o cruzamento H2H funcione corretamente.
        """
        try:
            data = self.api.get_partidas(rodada)
            conn = sqlite3.connect(self.db_path)
            partidas = data.get('partidas', [])

            for partida in partidas:
                casa_id      = partida.get('clube_casa_id')
                visitante_id = partida.get('clube_visitante_id')
                placar_casa  = partida.get('placar_oficial_mandante')
                placar_visit = partida.get('placar_oficial_visitante')
                aprov_m = self._calcular_aproveitamento(partida.get('aproveitamento_mandante'))
                aprov_v = self._calcular_aproveitamento(partida.get('aproveitamento_visitante'))

                conn.execute("""
                    INSERT OR REPLACE INTO partidas
                    (partida_id, rodada,
                     clube_casa_id, clube_visitante_id,
                     clube_id_a, clube_id_b,
                     placar_oficial_mandante, placar_oficial_visitante,
                     placar_a, placar_b,
                     aproveitamento_mandante, aproveitamento_visitante, valid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    partida.get('partida_id'), rodada,
                    casa_id, visitante_id,
                    casa_id, visitante_id,       # clube_id_a / clube_id_b (alias para H2H)
                    placar_casa, placar_visit,
                    placar_casa, placar_visit,   # placar_a / placar_b (alias para H2H)
                    aprov_m, aprov_v,
                    partida.get('valid')
                ))

            conn.commit()
            conn.close()
            logger.info(f"⚽ {len(partidas)} partidas coletadas (com placar H2H) - Rodada {rodada}")
        except Exception as e:
            logger.error(f"❌ Erro ao coletar partidas: {e}")
            raise

    @staticmethod
    def _calcular_aproveitamento(aproveitamento) -> float:
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
        conn = sqlite3.connect(self.db_path)
        if ultimas_n_rodadas:
            df = pd.read_sql_query(
                "SELECT * FROM pontuacoes WHERE atleta_id=? ORDER BY rodada DESC LIMIT ?",
                conn, params=(atleta_id, ultimas_n_rodadas)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM pontuacoes WHERE atleta_id=? ORDER BY rodada DESC",
                conn, params=(atleta_id,)
            )
        conn.close()
        return df
