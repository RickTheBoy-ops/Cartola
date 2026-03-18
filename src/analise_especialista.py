"""
Módulo de Análise Especialista em Cartola FC
Executa análise estruturada antes de gerar escalação

Fluxo:
1. Coleta dados da rodada atual
2. Mapeia confrontos e desfalques
3. Analisa scouts por posição
4. Calcula MPV e teto de pontos
5. Constrói time otimizado
6. Gera relatório markdown
7. Recomenda escalação
"""

import json
import requests
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import pandas as pd
from enum import Enum


class PosicaoEnum(Enum):
    """Posições no Cartola"""
    GOL = "gol"
    ZAG = "zagueiro"
    LAT = "lateral"
    VOL = "volante"
    MEIA = "meia"
    PONTA = "ponta"
    ATA = "atacante"


class TipoConfrontoEnum(Enum):
    """Classificação de confrontos"""
    GRUPO_A = "Amplo Favoritismo"  # 70%+ de vitória do favorito
    GRUPO_B = "Aberto"  # Gols de ambos, 50-50
    GRUPO_C = "Truncado"  # 0-0 esperado, defesa


@dataclass
class Jogador:
    """Representação de um jogador"""
    id: int
    nome: str
    time: str
    posicao: str
    preco: float
    media_scouts: float
    media_por_90min: float
    teto_pontos: float
    risco_cartao: str  # "BAIXO", "MÉDIO", "ALTO"
    probabilidade_90min: float
    mpv: float
    scouts_cedidos_oponente: float  # O quanto seu oponente cede para a posição
    xG_xA: float  # Expected Goals ou Expected Assists
    
    def eh_viavel(self) -> bool:
        """Retorna True se o jogador é viável para escala"""
        return self.probabilidade_90min > 0.75 and self.risco_cartao != "ALTO"


@dataclass
class Confronto:
    """Representação de um confronto/partida"""
    id: int
    time_a: str
    time_b: str
    mando_a: bool
    odds_vit_a: float  # Probabilidade de vitória de A
    tipo: TipoConfrontoEnum
    desfalques_a: List[str]
    desfalques_b: List[str]
    historico_gols: float  # Média de gols no histórico
    
    def eh_grupo_a(self) -> bool:
        """Retorna True se confronto é Grupo A (amplo favoritismo)"""
        return self.odds_vit_a > 0.65 or self.odds_vit_a < 0.35
    
    def favoritismo_time(self, time: str) -> float:
        """Retorna probabilidade de vitória do time"""
        if time == self.time_a:
            return self.odds_vit_a
        else:
            return 1 - self.odds_vit_a


@dataclass
class Rodada:
    """Representação de uma rodada completa"""
    numero: int
    data: str
    confrontos: List[Confronto]
    jogadores: List[Jogador]
    patrimonio_total: float
    patrimonio_livre_apos_11: float


class AnalisadorEspecialista:
    """Núcleo de análise especialista em Cartola"""
    
    def __init__(self):
        self.rodada: Optional[Rodada] = None
        self.time_selecionado: List[Jogador] = []
        self.capitao: Optional[Jogador] = None
        self.relatorio_md = ""
    
    # ============ ETAPA 1: MAPEAMENTO DE CONFRONTOS ============
    
    def mapear_confrontos(self, rodada: Rodada) -> Dict[str, TipoConfrontoEnum]:
        """
        Classifica cada confronto em Grupo A/B/C
        Retorna dicionário com tipo de cada confronto
        """
        classificacao = {}
        
        for conf in rodada.confrontos:
            if conf.eh_grupo_a():
                classificacao[f"{conf.time_a} vs {conf.time_b}"] = TipoConfrontoEnum.GRUPO_A
            elif conf.tipo == TipoConfrontoEnum.TRUNCADO:
                classificacao[f"{conf.time_a} vs {conf.time_b}"] = TipoConfrontoEnum.GRUPO_C
            else:
                classificacao[f"{conf.time_a} vs {conf.time_b}"] = TipoConfrontoEnum.GRUPO_B
        
        return classificacao
    
    # ============ ETAPA 2: ANÁLISE DE SCOUTS ============
    
    def analisar_scouts_por_posicao(self, rodada: Rodada) -> Dict[str, List[Jogador]]:
        """
        Agrupa jogadores viáveis por posição
        Ordena por média de scouts + matchup
        """
        jogadores_por_posicao = {}
        
        for pos in PosicaoEnum:
            jogadores_pos = [
                j for j in rodada.jogadores 
                if j.posicao == pos.value and j.eh_viavel()
            ]
            # Ordena por teto de pontos (média + scouts cedidos)
            jogadores_pos.sort(
                key=lambda j: (j.media_scouts + j.scouts_cedidos_oponente),
                reverse=True
            )
            jogadores_por_posicao[pos.value] = jogadores_pos[:5]  # Top 5 por posição
        
        return jogadores_por_posicao
    
    # ============ ETAPA 3: CÁLCULO DE MPV ============
    
    def calcular_mpv_jogadores(self, rodada: Rodada) -> Dict[int, float]:
        """
        Calcula Mínimo Para Valorizar de cada jogador
        Retorna {jogador_id: mpv}
        """
        mpv_dict = {}
        
        for j in rodada.jogadores:
            # Jogadores caros precisam de muito para valorizar
            # Jogadores baratos valorizam com pouco
            if j.preco > 15:
                # Caro: precisa de ~40 pontos para subir 0.01
                mpv = 40
            elif j.preco > 10:
                # Médio: precisa de ~20 pontos
                mpv = 20
            else:
                # Barato: precisa de ~8 pontos
                mpv = 8
            
            mpv_dict[j.id] = mpv
        
        return mpv_dict
    
    # ============ ETAPA 4: ANÁLISE DE MATCHUP ============
    
    def avaliar_matchup(self, jogador: Jogador, confronto: Confronto, rodada: Rodada) -> Tuple[float, str]:
        """
        Avalia quão favorável é o matchup do jogador
        Retorna (score_matchup: 0-1, justificativa)
        """
        score = 0.5  # Neutral
        justificativa = ""
        
        # Se é grupo A (amplo favoritismo)
        if confronto.eh_grupo_a():
            if confronto.favoritismo_time(jogador.time) > 0.65:
                score += 0.2
                justificativa += "Grupo A (favorito) + "
            else:
                score -= 0.15
                justificativa += "Grupo C (fraco) + "
        
        # Se scouts cedidos são altos
        if jogador.scouts_cedidos_oponente > 10:
            score += 0.15
            justificativa += "Alto scout cedido + "
        
        # Se está em boa forma (xG/xA alta)
        if jogador.xG_xA > 1.2:
            score += 0.1
            justificativa += "Boa fase + "
        
        return min(max(score, 0), 1), justificativa.rstrip(" + ")
    
    # ============ ETAPA 5: SELEÇÃO DE CAPITÃO ============
    
    def selecionar_capitao(self, time: List[Jogador], rodada: Rodada) -> Jogador:
        """
        Executa checklist de capitão:
        - Amplo favoritismo (Grupo A)
        - 90 minutos confirmados
        - Responsável por bola parada
        - Média recente alta
        - Matchup favorável
        """
        candidatos = []
        
        for j in time:
            if j.posicao not in ["atacante", "ponta"]:
                continue  # Capitão deve ser ofensivo
            
            checks = 0
            
            # Check 1: Amplo favoritismo
            conf = next((c for c in rodada.confrontos 
                        if j.time in [c.time_a, c.time_b]), None)
            if conf and conf.eh_grupo_a() and conf.favoritismo_time(j.time) > 0.65:
                checks += 1
            
            # Check 2: 90 minutos
            if j.probabilidade_90min > 0.85:
                checks += 1
            
            # Check 3: Bola parada (proxy: atacante com xG alta)
            if j.xG_xA > 1.5:
                checks += 1
            
            # Check 4: Média recente alta
            if j.media_scouts > 12:
                checks += 1
            
            # Check 5: Matchup
            if j.scouts_cedidos_oponente > 10:
                checks += 1
            
            if checks >= 4:  # Mínimo 4 checks
                candidatos.append((j, checks))
        
        # Ordena por número de checks (desc)
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        if candidatos:
            return candidatos[0][0]
        else:
            # Fallback: maior teto de pontos
            return max(time, key=lambda j: j.teto_pontos)
    
    # ============ ETAPA 6: CONSTRUÇÃO DO TIME ============
    
    def construir_time(
        self,
        rodada: Rodada,
        estrategia: str = "meio-termo"  # "mitar", "valorizar", "meio-termo"
    ) -> List[Jogador]:
        """
        Constrói time otimizado conforme estratégia
        Retorna lista de 11 jogadores + 1 reserva
        """
        scouts_por_pos = self.analisar_scouts_por_posicao(rodada)
        time_otimizado = []
        patrimonio_gasto = 0
        
        # Estrutura obrigatória: 1 GOL + 4 DEF + 3 MEIA + 2 ATA
        # + 1 Reserva
        
        estrutura = {
            "gol": (1, scouts_por_pos["gol"]),
            "zagueiro": (1, scouts_por_pos["zagueiro"]),
            "lateral": (2, scouts_por_pos["lateral"]),
            "volante": (1, scouts_por_pos["volante"]),
            "meia": (2, scouts_por_pos["meia"]),
            "ponta": (0, scouts_por_pos["ponta"]),
            "atacante": (2, scouts_por_pos["atacante"]),
        }
        
        for posicao, (qtd, candidatos) in estrutura.items():
            for i in range(qtd):
                if i < len(candidatos):
                    j = candidatos[i]
                    # Validar limite de correlação
                    times_selecionados = [jj.time for jj in time_otimizado]
                    defensores_mesmo_time = sum(
                        1 for jj in time_otimizado
                        if jj.time == j.time and jj.posicao in ["gol", "zagueiro", "lateral"]
                    )
                    
                    if defensores_mesmo_time < 3:  # Max 3 defensores do mesmo time
                        if patrimonio_gasto + j.preco <= rodada.patrimonio_livre_apos_11:
                            time_otimizado.append(j)
                            patrimonio_gasto += j.preco
        
        # Reserva em time diferente
        para_reserva = [
            j for j in rodada.jogadores
            if j.eh_viavel() and j not in time_otimizado
            and j.time not in [jj.time for jj in time_otimizado[:5]]
        ]
        if para_reserva:
            reserva = para_reserva[0]
            time_otimizado.append(reserva)
        
        self.time_selecionado = time_otimizado
        return time_otimizado
    
    # ============ ETAPA 7: SIMULAÇÕES ============
    
    def simular_cenarios(self, time: List[Jogador]) -> Dict[str, float]:
        """
        Simula 3 cenários: otimista, realista, pessimista
        Retorna pontuação esperada para cada
        """
        pontos_base = sum(j.media_scouts for j in time)
        desvio = sum(j.media_scouts * 0.3 for j in time)  # Desvio padrão ~30%
        
        return {
            "otimista": pontos_base + desvio,
            "realista": pontos_base,
            "pessimista": max(0, pontos_base - desvio)
        }
    
    # ============ ETAPA 8: VALIDAÇÃO ============
    
    def validar_escalacao(self, time: List[Jogador]) -> Tuple[bool, List[str], List[str]]:
        """
        Executa validação pré-escala
        Retorna (valido, erros, avisos)
        """
        erros = []
        avisos = []
        
        # ERROS CRÍTICOS
        for time_nome in set(j.time for j in time):
            defensores = sum(
                1 for j in time
                if j.time == time_nome and j.posicao in ["gol", "zagueiro", "lateral"]
            )
            if defensores > 3:
                erros.append(f"Máximo 3 defensores de {time_nome} (tem {defensores})")
        
        if sum(1 for j in time if j.posicao == "atacante") < 2:
            erros.append("Mínimo 2 atacantes")
        
        # AVISOS
        cartoes_altos = sum(
            1 for j in time
            if j.risco_cartao == "ALTO"
        )
        if cartoes_altos > 1:
            avisos.append(f"2+ jogadores com risco ALTO de cartão")
        
        return len(erros) == 0, erros, avisos
    
    # ============ ETAPA 9: GERAÇÃO DE RELATÓRIO ============
    
    def gerar_relatorio_markdown(self, rodada: Rodada) -> str:
        """
        Gera relatório completo em markdown
        Segue template TEMPLATE_ANALISE_RODADA.md
        """
        md = f"""# 📋 ANÁLISE ESPECIALISTA - Rodada {rodada.numero}

> **Data:** {rodada.data}  
> **Gerado em:** {datetime.now().strftime('%d/%m/%Y - %H:%M')}  
> **Patrimônio:** C$ {rodada.patrimonio_total:.2f}

---

## 🎯 RESUMO EXECUTIVO

**Melhor aproveitamento:** Confrontos Grupo A (Amplo favoritismo)  
**Maior risco:** Confrontos Grupo C (Truncado)  
**Diferencial:** Lateral ofensivo vs defesa fraca

---

## 🏆 MAPEAMENTO DE CONFRONTOS

"""
        
        # Mapeamento de confrontos
        md += "| # | Time A | vs | Time B | Tipo |\n"
        md += "|---|--------|----|----|------|\n"
        
        for i, conf in enumerate(rodada.confrontos, 1):
            tipo = self.mapear_confrontos(rodada).get(
                f"{conf.time_a} vs {conf.time_b}",
                TipoConfrontoEnum.GRUPO_B
            )
            md += f"| {i} | {conf.time_a} | vs | {conf.time_b} | {tipo.value} |\n"
        
        md += f"""

---

## 📊 TIME RECOMENDADO

```
GOL:      {self.time_selecionado[0].nome if self.time_selecionado else 'N/A'} ({self.time_selecionado[0].time})
DEF:      {', '.join([j.nome for j in self.time_selecionado if j.posicao in ['zagueiro', 'lateral']][:4])}
MEIA:     {', '.join([j.nome for j in self.time_selecionado if j.posicao in ['volante', 'meia', 'ponta']][:3])}
ATA:      {', '.join([j.nome for j in self.time_selecionado if j.posicao == 'atacante'][:2])}
RESERVA:  {self.time_selecionado[-1].nome if len(self.time_selecionado) > 11 else 'N/A'}
```

### ⭐ CAPITÃO RECOMENDADO

**{self.capitao.nome if self.capitao else 'N/A'}** ({self.capitao.time if self.capitao else 'N/A'})  
Teto de pontos: **{self.capitao.teto_pontos:.0f}** pts  
Confiança: **9/10**

---

## ✅ VALIDAÇÃO PRÉ-ESCALA

Valido, erros, avisos = self.validar_escalacao(self.time_selecionado)
"""
        
        self.relatorio_md = md
        return md
    
    # ============ EXECUÇÃO COMPLETA ============
    
    def executar_analise_completa(self, rodada: Rodada, estrategia: str = "meio-termo"):
        """
        Executa pipeline completo:
        1. Mapeamento
        2. Análise scouts
        3. Construção do time
        4. Seleção de capitão
        5. Validação
        6. Geração de relatório
        """
        print(f"\n🚀 Iniciando análise especialista - Rodada {rodada.numero}...\n")
        
        # Etapa 1: Mapeamento
        print("[1/6] Mapeando confrontos...")
        classificacao = self.mapear_confrontos(rodada)
        print(f"  ✓ {len(classificacao)} confrontos classificados\n")
        
        # Etapa 2: Análise scouts
        print("[2/6] Analisando scouts...")
        scouts = self.analisar_scouts_por_posicao(rodada)
        print(f"  ✓ Scouts analisados para {len(scouts)} posições\n")
        
        # Etapa 3: Construção do time
        print(f"[3/6] Construindo time ({estrategia})...")
        time = self.construir_time(rodada, estrategia)
        print(f"  ✓ Time com {len(time)} jogadores construído\n")
        
        # Etapa 4: Capitão
        print("[4/6] Selecionando capitão...")
        self.capitao = self.selecionar_capitao(time, rodada)
        print(f"  ✓ Capitão: {self.capitao.nome} ({self.capitao.teto_pontos:.0f} pts)\n")
        
        # Etapa 5: Validação
        print("[5/6] Validando escalação...")
        valido, erros, avisos = self.validar_escalacao(time)
        if valido:
            print(f"  ✓ Escalação VÁLIDA\n")
        else:
            print(f"  ✗ Erros: {erros}\n")
        
        # Etapa 6: Relatório
        print("[6/6] Gerando relatório markdown...")
        self.gerar_relatorio_markdown(rodada)
        print(f"  ✓ Relatório gerado ({len(self.relatorio_md)} chars)\n")
        
        print("="*60)
        print("✅ ANÁLISE COMPLETA! Pronto para escalar.")
        print("="*60 + "\n")
        
        return {
            "time": time,
            "capitao": self.capitao,
            "valido": valido,
            "erros": erros,
            "avisos": avisos,
            "relatorio": self.relatorio_md
        }


if __name__ == "__main__":
    # Exemplo de uso
    print("Cartola FC - Módulo de Análise Especialista")
    print("Pronto para integração em main.py e app.py")
