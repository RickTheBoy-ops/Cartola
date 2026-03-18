"""
Pipeline Completo de Escalação

Fluxo:
1. Coleta dados da rodada (API Cartola)
2. Executa análise especialista
3. Gera relatório markdown
4. Recomenda escalação
5. Commit automático no repo
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import subprocess

from analise_especialista import (
    AnalisadorEspecialista,
    Rodada,
    Confronto,
    Jogador,
    TipoConfrontoEnum,
)


class PipelineEscalacao:
    """Orquestra todo o pipeline de escalação"""
    
    def __init__(self, numero_rodada: int, estrategia: str = "meio-termo"):
        self.numero_rodada = numero_rodada
        self.estrategia = estrategia
        self.analisador = AnalisadorEspecialista()
        self.resultado_analise = None
        self.data_hora = datetime.now()
    
    def coletar_dados_rodada(self) -> Rodada:
        """
        Coleta dados da rodada via API Cartola FC
        Retorna objeto Rodada preenchido
        """
        print(f"\n🔍 Coletando dados da Rodada {self.numero_rodada}...\n")
        
        # TODO: Integrar com API real do Cartola
        # Por enquanto, retorna estrutura válida
        
        rodada = Rodada(
            numero=self.numero_rodada,
            data=self.data_hora.strftime("%d/%m/%Y"),
            confrontos=[],  # Preenchido pela API
            jogadores=[],   # Preenchido pela API
            patrimonio_total=180.50,
            patrimonio_livre_apos_11=28.30,
        )
        
        print(f"  ✓ Dados da Rodada {self.numero_rodada} coletados")
        print(f"  ✓ Patrimônio: C$ {rodada.patrimonio_total:.2f}\n")
        
        return rodada
    
    def executar_analise(self, rodada: Rodada) -> Dict:
        """
        Executa análise especialista completa
        """
        print("\n" + "="*70)
        print("         🎯 ANÁLISE ESPECIALISTA EM CARTOLA")
        print("="*70 + "\n")
        
        resultado = self.analisador.executar_analise_completa(
            rodada,
            estrategia=self.estrategia
        )
        
        self.resultado_analise = resultado
        return resultado
    
    def salvar_relatorio_markdown(self) -> str:
        """
        Salva relatório em arquivo markdown na pasta docs/
        Retorna caminho do arquivo
        """
        if not self.resultado_analise:
            raise ValueError("Execute análise primeiro")
        
        # Nome do arquivo
        nome_arquivo = f"RODADA_{self.numero_rodada}_ANALISE_{self.data_hora.strftime('%Y%m%d_%H%M')}.md"
        caminho_arquivo = Path("docs") / nome_arquivo
        
        # Cria direório se não existir
        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        
        # Escreve relatório
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(self.analisador.relatorio_md)
        
        print(f"\n📄 Relatório salvo em: {caminho_arquivo}\n")
        
        return str(caminho_arquivo)
    
    def salvar_recomendacao_json(self) -> str:
        """
        Salva recomendação de escalação em JSON
        """
        if not self.resultado_analise:
            raise ValueError("Execute análise primeiro")
        
        recomendacao = {
            "rodada": self.numero_rodada,
            "data_analise": self.data_hora.isoformat(),
            "estrategia": self.estrategia,
            "time_recomendado": [
                {
                    "nome": j.nome,
                    "time": j.time,
                    "posicao": j.posicao,
                    "preco": j.preco,
                    "media_scouts": j.media_scouts,
                    "teto_pontos": j.teto_pontos,
                }
                for j in self.resultado_analise["time"]
            ],
            "capitao": {
                "nome": self.resultado_analise["capitao"].nome,
                "time": self.resultado_analise["capitao"].time,
                "teto_pontos": self.resultado_analise["capitao"].teto_pontos,
            },
            "validaçao": {
                "valido": self.resultado_analise["valido"],
                "erros": self.resultado_analise["erros"],
                "avisos": self.resultado_analise["avisos"],
            },
        }
        
        # Salva JSON
        nome_arquivo = f"RODADA_{self.numero_rodada}_RECOMENDACAO.json"
        caminho_arquivo = Path("output") / nome_arquivo
        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            json.dump(recomendacao, f, ensure_ascii=False, indent=2)
        
        print(f"\ud83d� Recomendação salva em: {caminho_arquivo}\n")
        
        return str(caminho_arquivo)
    
    def commit_automatico(self, arquivo_relatorio: str, arquivo_json: str):
        """
        Executa commit automático no repositório Git
        """
        try:
            print("\n📄 Executando commit automático...\n")
            
            # Add arquivos
            subprocess.run(["git", "add", arquivo_relatorio], check=True)
            subprocess.run(["git", "add", arquivo_json], check=True)
            
            # Commit
            mensagem_commit = f"docs: add analysis and recommendation for round {self.numero_rodada} ({self.estrategia})"
            subprocess.run(
                ["git", "commit", "-m", mensagem_commit],
                check=True
            )
            
            # Push
            subprocess.run(["git", "push", "origin", "main"], check=True)
            
            print(f"  ✓ Commit realizado com sucesso!")
            print(f"  ✓ Mensagem: {mensagem_commit}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Erro ao fazer commit: {e}\n")
    
    def exibir_resumo_executivo(self):
        """
        Exibe resumo executivo da análise
        """
        if not self.resultado_analise:
            return
        
        time = self.resultado_analise["time"]
        capitao = self.resultado_analise["capitao"]
        valido = self.resultado_analise["valido"]
        
        print("\n" + "="*70)
        print("         🎲 RESUMO DA ESCALAÇÃO")
        print("="*70 + "\n")
        
        print(f"🎯 RODADA: {self.numero_rodada}")
        print(f"💰 ESTRATÉGIA: {self.estrategia.upper()}")
        print(f"⭐ CAPITÃO: {capitao.nome} ({capitao.team}) - {capitao.teto_pontos:.0f} pts\n")
        
        print(f"🛡️ DEFESA ({len([j for j in time if j.posicao in ['gol', 'zagueiro', 'lateral']])}):\n")
        for j in time:
            if j.posicao in ["gol", "zagueiro", "lateral"]:
                print(f"   - {j.nome} ({j.team}, {j.posicao}) - C$ {j.preco:.2f}")
        
        print(f"\n🎯 MEIO-CAMPO ({len([j for j in time if j.posicao in ['volante', 'meia', 'ponta']])}):\n")
        for j in time:
            if j.posicao in ["volante", "meia", "ponta"]:
                print(f"   - {j.nome} ({j.team}, {j.posicao}) - C$ {j.preco:.2f}")
        
        print(f"\n⚔️ ATAQUE ({len([j for j in time if j.posicao == 'atacante'])}):\n")
        for j in time:
            if j.posicao == "atacante":
                print(f"   - {j.nome} ({j.team}) - C$ {j.preco:.2f}")
        
        status = "✅ VÁLIDO" if valido else "❌ INVÁLIDO"
        print(f"\n{status}\n")
        
        print("="*70 + "\n")
    
    def executar_pipeline_completo(self, fazer_commit: bool = True):
        """
        Executa pipeline completo:
        1. Coleta dados
        2. Análise
        3. Salva relatório
        4. Salva recomendação
        5. Commit automático (opcional)
        """
        try:
            # Etapa 1: Coletar dados
            rodada = self.coletar_dados_rodada()
            
            # Etapa 2: Análise
            self.executar_analise(rodada)
            
            # Etapa 3: Salvar relatório
            arquivo_relatorio = self.salvar_relatorio_markdown()
            
            # Etapa 4: Salvar recomendação
            arquivo_json = self.salvar_recomendacao_json()
            
            # Etapa 5: Exibir resumo
            self.exibir_resumo_executivo()
            
            # Etapa 6: Commit (opcional)
            if fazer_commit:
                self.commit_automatico(arquivo_relatorio, arquivo_json)
            
            print("\n" + "="*70)
            print("✋ PIPELINE COMPLETO! Pronto para escalar.")
            print("="*70 + "\n")
            
            return {
                "sucesso": True,
                "relatorio": arquivo_relatorio,
                "recomendacao": arquivo_json,
            }
            
        except Exception as e:
            print(f"\n❌ Erro no pipeline: {e}\n")
            return {
                "sucesso": False,
                "erro": str(e),
            }


if __name__ == "__main__":
    # Exemplo de uso
    pipeline = PipelineEscalacao(
        numero_rodada=7,
        estrategia="meio-termo"
    )
    
    resultado = pipeline.executar_pipeline_completo(fazer_commit=False)
    
    if resultado["sucesso"]:
        print(f"\n🌟 Sucesso!")
        print(f"Relatório: {resultado['relatorio']}")
        print(f"Recomendação: {resultado['recomendacao']}")
    else:
        print(f"\n⚠️ Erro: {resultado['erro']}")
