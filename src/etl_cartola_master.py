"""
Sistema ETL - Dataset Mestre do Cartola FC
===========================================

Engenharia de Dados Sênior - Pipeline completo de ETL
Extrai dados de 3 APIs oficiais, enriquece com cruzamento de informações
e gera um dataset mestre pronto para análise de IA.

Autor: Engenheiro de Dados Sênior
Data: 2026-01-28
Versão: 1.0
"""

# ============================================================================
# IMPORTAÇÕES
# ============================================================================
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
class Config:
    """Configurações da API do Cartola FC"""
    
    # Endpoints da API oficial
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    # Mapeamento de posições
    POSICOES = {
        1: "Goleiro",
        2: "Lateral",
        3: "Zagueiro",
        4: "Meia",
        5: "Atacante",
        6: "Técnico"
    }
    
    # Status dos atletas
    STATUS_PROVAVEL = 7
    
    # Headers para requisições
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Arquivo de saída
    ARQUIVO_SAIDA = "tabela_completa_cartola.xlsx"


# ============================================================================
# CLASSE PRINCIPAL - ETL CARTOLA FC
# ============================================================================
class CartolaETL:
    """
    Pipeline ETL completo para processar dados do Cartola FC
    
    Fases:
    1. Extract: Extrai dados de 3 APIs
    2. Transform: Enriquece e cruza informações
    3. Load: Gera arquivo Excel formatado
    """
    
    def __init__(self):
        """Inicializa o pipeline ETL"""
        self.dados_atletas = None
        self.dados_clubes = None
        self.dados_partidas = None
        
        self.mapa_clubes = {}
        self.mapa_escudos = {}
        self.mapa_partidas = {}
        
        self.df_final = None
        
        print("="*80)
        print("🏆 SISTEMA ETL - CARTOLA FC - DATASET MESTRE")
        print("="*80)
        print()
    
    # ========================================================================
    # FASE 1: EXTRACT (Extração de Dados)
    # ========================================================================
    
    def extrair_dados_api(self, url: str, nome: str) -> Optional[dict]:
        """
        Extrai dados de um endpoint da API
        
        Args:
            url (str): URL do endpoint
            nome (str): Nome descritivo do endpoint (para logs)
            
        Returns:
            Optional[dict]: Dados em formato JSON ou None se falhar
        """
        try:
            print(f"📡 Extraindo dados de: {nome}...")
            print(f"   URL: {url}")
            
            response = requests.get(url, headers=Config.HEADERS, timeout=30)
            response.raise_for_status()
            
            dados = response.json()
            
            print(f"   ✅ Dados extraídos com sucesso!")
            print(f"   📊 Tamanho da resposta: {len(str(dados)):,} caracteres")
            
            return dados
            
        except requests.RequestException as e:
            print(f"   ❌ Erro na requisição: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"   ❌ Erro ao decodificar JSON: {e}")
            return None
        except Exception as e:
            print(f"   ❌ Erro inesperado: {e}")
            return None
    
    def extract(self) -> bool:
        """
        Executa a fase de extração de todas as APIs
        
        Returns:
            bool: True se todas as extrações foram bem-sucedidas
        """
        print("\n" + "="*80)
        print("FASE 1: EXTRACT - Extração de Dados das APIs")
        print("="*80 + "\n")
        
        # Extrai atletas
        self.dados_atletas = self.extrair_dados_api(
            Config.API_ATLETAS,
            "Atletas do Mercado"
        )
        
        # Extrai clubes
        self.dados_clubes = self.extrair_dados_api(
            Config.API_CLUBES,
            "Clubes"
        )
        
        # Extrai partidas
        self.dados_partidas = self.extrair_dados_api(
            Config.API_PARTIDAS,
            "Partidas da Rodada"
        )
        
        # Valida se todas as extrações foram bem-sucedidas
        if not all([self.dados_atletas, self.dados_clubes, self.dados_partidas]):
            print("\n❌ Falha na extração de dados!")
            return False
        
        print(f"\n✅ Fase de EXTRAÇÃO concluída com sucesso!")
        return True
    
    # ========================================================================
    # FASE 2: TRANSFORM (Transformação e Enriquecimento)
    # ========================================================================
    
    def criar_dicionarios_clubes(self) -> bool:
        """
        Passo A: Cria mapas de clubes (id -> nome, id -> escudo)
        
        Returns:
            bool: True se criou com sucesso
        """
        try:
            print("\n🔄 Passo A: Criando dicionários de clubes...")
            
            # Extrai informações dos clubes
            for clube_id, clube_info in self.dados_clubes.items():
                clube_id_int = int(clube_id)
                self.mapa_clubes[clube_id_int] = clube_info.get('nome', 'Desconhecido')
                self.mapa_escudos[clube_id_int] = clube_info.get('escudos', {})
            
            print(f"   ✅ {len(self.mapa_clubes)} clubes mapeados:")
            for clube_id, nome in sorted(self.mapa_clubes.items()):
                print(f"      • {clube_id}: {nome}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro ao criar dicionários: {e}")
            return False
    
    def enriquecer_partidas(self) -> bool:
        """
        Passo B: Analisa partidas e cria mapa de adversários e mando de campo
        
        Estrutura do mapa:
        {
            clube_id: {
                'adversario': nome_do_adversario,
                'adversario_id': id_do_adversario,
                'mando_campo': 'Casa' ou 'Fora',
                'local': nome_do_estadio
            }
        }
        
        Returns:
            bool: True se processou com sucesso
        """
        try:
            print("\n🔄 Passo B: Enriquecendo dados de partidas...")
            
            total_partidas = 0
            
            # A API pode retornar as partidas em formatos diferentes
            # Tenta extrair lista de partidas da estrutura
            partidas_raw = self.dados_partidas
            
            # Se for dict com chave 'partidas'
            if isinstance(partidas_raw, dict):
                if 'partidas' in partidas_raw:
                    partidas_lista = partidas_raw['partidas']
                else:
                    # Pode ser um dict onde as chaves são rodadas
                    # Precisamos iterar pelos valores
                    partidas_lista = []
                    for key, value in partidas_raw.items():
                        if isinstance(value, dict) and 'partidas' in value:
                            partidas_lista.extend(value['partidas'])
                        elif isinstance(value, list):
                            partidas_lista.extend(value)
                        elif isinstance(value, dict) and 'clube_casa_id' in value:
                            partidas_lista.append(value)
            elif isinstance(partidas_raw, list):
                partidas_lista = partidas_raw
            else:
                print(f"   ⚠️ Formato inesperado de partidas: {type(partidas_raw)}")
                partidas_lista = []
            
            # Itera sobre as partidas
            for partida in partidas_lista:
                if not isinstance(partida, dict):
                    continue
                    
                total_partidas += 1
                
                # Extrai informações da partida
                clube_casa_id = partida.get('clube_casa_id')
                clube_visitante_id = partida.get('clube_visitante_id')
                local = partida.get('local', 'Local não informado')
                
                if not clube_casa_id or not clube_visitante_id:
                    continue
                
                # Nome dos clubes
                nome_casa = self.mapa_clubes.get(clube_casa_id, f"Clube {clube_casa_id}")
                nome_visitante = self.mapa_clubes.get(clube_visitante_id, f"Clube {clube_visitante_id}")
                
                # Mapa para o clube da casa (mandante)
                self.mapa_partidas[clube_casa_id] = {
                    'adversario': nome_visitante,
                    'adversario_id': clube_visitante_id,
                    'mando_campo': 'Casa',
                    'local': local,
                    'confronto': f"{nome_casa} x {nome_visitante}"
                }
                
                # Mapa para o clube visitante
                self.mapa_partidas[clube_visitante_id] = {
                    'adversario': nome_casa,
                    'adversario_id': clube_casa_id,
                    'mando_campo': 'Fora',
                    'local': local,
                    'confronto': f"{nome_casa} x {nome_visitante}"
                }
            
            print(f"   ✅ {total_partidas} partidas processadas")
            print(f"   ✅ {len(self.mapa_partidas)} clubes com informações de partida")
            
            # Mostra alguns exemplos
            if self.mapa_partidas:
                print(f"\n   📋 Exemplos de partidas:")
                count = 0
                for clube_id, info in self.mapa_partidas.items():
                    if count >= 3:
                        break
                    clube_nome = self.mapa_clubes.get(clube_id, f"Clube {clube_id}")
                    print(f"      • {clube_nome}: {info['confronto']} ({info['mando_campo']})")
                    count += 1
            else:
                print(f"   ⚠️ Aviso: Nenhuma partida foi mapeada!")
                print(f"   ℹ️ Pode ser fora da temporada ou entre rodadas")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro ao enriquecer partidas: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def processar_atletas(self) -> bool:
        """
        Passo C: Processa atletas e cria DataFrame final
        
        Filtra apenas atletas com status_id == 7 (Provável)
        Cruza com informações de clubes e partidas
        
        Returns:
            bool: True se processou com sucesso
        """
        try:
            print("\n🔄 Passo C: Processando atletas e cruzando dados...")
            
            # Lista para armazenar dados dos atletas
            lista_atletas = []
            
            # Extrai lista de atletas
            atletas = self.dados_atletas.get('atletas', [])
            
            if isinstance(atletas, dict):
                # Se for um dicionário, converte para lista
                atletas = list(atletas.values())
            
            print(f"   📊 Total de atletas na API: {len(atletas)}")
            
            # Contadores
            total_provaveis = 0
            total_sem_partida = 0
            
            for atleta in atletas:
                # Filtro: apenas status_id == 7 (Provável)
                if atleta.get('status_id') != Config.STATUS_PROVAVEL:
                    continue
                
                total_provaveis += 1
                
                # Extrai dados básicos do atleta
                atleta_id = atleta.get('atleta_id')
                apelido = atleta.get('apelido', 'Desconhecido')
                posicao_id = atleta.get('posicao_id')
                posicao = Config.POSICOES.get(posicao_id, 'Desconhecida')
                clube_id = atleta.get('clube_id')
                clube = self.mapa_clubes.get(clube_id, 'Desconhecido')
                
                # Estatísticas
                preco = atleta.get('preco_num', 0.0)
                media = atleta.get('media_num', 0.0)
                ultima_pontuacao = atleta.get('pontos_num', 0.0)
                minimo_para_valorizar = atleta.get('minimo_para_valorizar', None)
                jogos_disputados = atleta.get('jogos_num', 0)
                
                # Busca informações da partida do clube
                info_partida = self.mapa_partidas.get(clube_id)
                
                if info_partida:
                    adversario = info_partida['adversario']
                    mando_campo = info_partida['mando_campo']
                    confronto = info_partida['confronto']
                else:
                    adversario = 'Sem jogo'
                    mando_campo = 'N/A'
                    confronto = 'Sem jogo'
                    total_sem_partida += 1
                
                # Cria registro do atleta
                registro = {
                    'atleta_id': atleta_id,
                    'apelido': apelido,
                    'posicao': posicao,
                    'clube': clube,
                    'adversario': adversario,
                    'mando_campo': mando_campo,
                    'confronto': confronto,
                    'preco': preco,
                    'media': media,
                    'ultima_pontuacao': ultima_pontuacao,
                    'minimo_para_valorizar': minimo_para_valorizar,
                    'jogos_disputados': jogos_disputados,
                    'media_num': media  # Mantém para compatibilidade
                }
                
                lista_atletas.append(registro)
            
            # Cria DataFrame
            self.df_final = pd.DataFrame(lista_atletas)
            
            # Ordena por média (decrescente)
            self.df_final = self.df_final.sort_values('media', ascending=False)
            self.df_final = self.df_final.reset_index(drop=True)
            
            print(f"\n   ✅ Processamento concluído!")
            print(f"   📊 Atletas prováveis (status_id=7): {total_provaveis}")
            print(f"   📊 Atletas no dataset final: {len(self.df_final)}")
            if total_sem_partida > 0:
                print(f"   ⚠️  Atletas sem partida nesta rodada: {total_sem_partida}")
            
            # Mostra estatísticas por posição
            print(f"\n   📈 Estatísticas por posição:")
            for posicao, count in self.df_final['posicao'].value_counts().items():
                print(f"      • {posicao}: {count} atletas")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro ao processar atletas: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def transform(self) -> bool:
        """
        Executa a fase completa de transformação
        
        Returns:
            bool: True se todas as transformações foram bem-sucedidas
        """
        print("\n" + "="*80)
        print("FASE 2: TRANSFORM - Transformação e Enriquecimento de Dados")
        print("="*80)
        
        # Passo A: Criar dicionários
        if not self.criar_dicionarios_clubes():
            return False
        
        # Passo B: Enriquecer partidas
        if not self.enriquecer_partidas():
            return False
        
        # Passo C: Processar atletas
        if not self.processar_atletas():
            return False
        
        print(f"\n✅ Fase de TRANSFORMAÇÃO concluída com sucesso!")
        return True
    
    # ========================================================================
    # FASE 3: LOAD (Exportação para Excel)
    # ========================================================================
    
    def load(self, arquivo_saida: str = None) -> bool:
        """
        Exporta o DataFrame final para arquivo Excel formatado
        
        Args:
            arquivo_saida (str): Nome do arquivo Excel (opcional)
            
        Returns:
            bool: True se exportou com sucesso
        """
        print("\n" + "="*80)
        print("FASE 3: LOAD - Exportação para Excel")
        print("="*80 + "\n")
        
        if self.df_final is None or self.df_final.empty:
            print("❌ Nenhum dado para exportar!")
            return False
        
        try:
            arquivo = arquivo_saida or Config.ARQUIVO_SAIDA
            
            print(f"💾 Exportando dados para: {arquivo}")
            print(f"   📊 Total de registros: {len(self.df_final)}")
            
            # Exporta para Excel com formatação
            with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
                self.df_final.to_excel(
                    writer,
                    index=False,
                    sheet_name='Dataset Cartola FC'
                )
                
                # Pega worksheet para formatação
                worksheet = writer.sheets['Dataset Cartola FC']
                
                # Ajusta largura das colunas
                for idx, col in enumerate(self.df_final.columns, 1):
                    max_length = max(
                        self.df_final[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    col_letter = chr(64 + idx)
                    worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
            
            print(f"   ✅ Arquivo criado com sucesso!")
            
            # Mostra prévia dos dados
            print(f"\n📋 Prévia dos TOP 10 atletas por média:")
            print(self.df_final[['apelido', 'posicao', 'clube', 'adversario', 
                                  'mando_campo', 'preco', 'media']].head(10).to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao exportar arquivo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # PIPELINE COMPLETO
    # ========================================================================
    
    def executar_pipeline(self) -> bool:
        """
        Executa o pipeline ETL completo
        
        Returns:
            bool: True se todo o pipeline foi executado com sucesso
        """
        inicio = datetime.now()
        
        print(f"🕐 Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # EXTRACT
        if not self.extract():
            print("\n❌ Pipeline interrompido na fase de EXTRACT")
            return False
        
        # TRANSFORM
        if not self.transform():
            print("\n❌ Pipeline interrompido na fase de TRANSFORM")
            return False
        
        # LOAD
        if not self.load():
            print("\n❌ Pipeline interrompido na fase de LOAD")
            return False
        
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()
        
        print("\n" + "="*80)
        print("🎉 PIPELINE ETL CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"🕐 Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duração: {duracao:.2f} segundos")
        print(f"📁 Arquivo gerado: {Config.ARQUIVO_SAIDA}")
        print(f"📊 Total de atletas: {len(self.df_final)}")
        print("="*80)
        
        return True


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================
def main():
    """Função principal - executa o pipeline ETL"""
    
    # Cria instância do ETL
    etl = CartolaETL()
    
    # Executa pipeline
    sucesso = etl.executar_pipeline()
    
    if sucesso:
        print("\n✅ Dataset mestre pronto para análise!")
        print("💡 Próximo passo: Use o arquivo Excel com uma IA para gerar a escalação ideal!")
    else:
        print("\n❌ Erro durante a execução do pipeline")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
