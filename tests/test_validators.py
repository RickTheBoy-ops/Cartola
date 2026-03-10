"""Testes para validadores Pydantic."""

import pytest
from pydantic import ValidationError
from src.models.validators import (
    AtletaModel,
    PartidaModel,
    MercadoStatusModel,
    PredicaoModel,
    EscalacaoModel,
    validar_atletas_batch
)


class TestAtletaModel:
    """Testes para AtletaModel."""
    
    def test_atleta_valido(self):
        """Testa criação de atleta válido."""
        atleta = AtletaModel(
            atleta_id=123,
            apelido="Neymar",
            clube_id=1,
            posicao_id=5,
            preco=25.50,
            pontos_rodada=10.5,
            media_pontos=8.75,
            jogos=5,
            status_id=7
        )
        
        assert atleta.atleta_id == 123
        assert atleta.apelido == "Neymar"
        assert atleta.preco == 25.50
    
    def test_preco_invalido_negativo(self):
        """Testa rejeição de preço negativo."""
        with pytest.raises(ValidationError) as exc_info:
            AtletaModel(
                atleta_id=1,
                apelido="Test",
                clube_id=1,
                posicao_id=5,
                preco=-5.0
            )
        
        assert "preco" in str(exc_info.value)
    
    def test_preco_fora_range(self):
        """Testa rejeição de preço fora do range."""
        with pytest.raises(ValidationError):
            AtletaModel(
                atleta_id=1,
                apelido="Test",
                clube_id=1,
                posicao_id=5,
                preco=150.0  # Muito alto
            )
    
    def test_apelido_vazio(self):
        """Testa rejeição de apelido vazio."""
        with pytest.raises(ValidationError):
            AtletaModel(
                atleta_id=1,
                apelido="",
                clube_id=1,
                posicao_id=5,
                preco=10.0
            )
    
    def test_posicao_invalida(self):
        """Testa rejeição de posição inválida."""
        with pytest.raises(ValidationError):
            AtletaModel(
                atleta_id=1,
                apelido="Test",
                clube_id=1,
                posicao_id=10,  # Posição não existe
                preco=10.0
            )
    
    def test_apelido_com_espacos(self):
        """Testa limpeza de espaços no apelido."""
        atleta = AtletaModel(
            atleta_id=1,
            apelido="  Neymar  ",
            clube_id=1,
            posicao_id=5,
            preco=25.0
        )
        
        assert atleta.apelido == "Neymar"


class TestPartidaModel:
    """Testes para PartidaModel."""
    
    def test_partida_valida(self):
        """Testa criação de partida válida."""
        partida = PartidaModel(
            partida_id=1,
            clube_casa_id=1,
            clube_visitante_id=2,
            rodada=10,
            placar_casa=2,
            placar_visitante=1
        )
        
        assert partida.clube_casa_id == 1
        assert partida.clube_visitante_id == 2
    
    def test_clubes_iguais(self):
        """Testa rejeição de clubes iguais."""
        with pytest.raises(ValidationError) as exc_info:
            PartidaModel(
                partida_id=1,
                clube_casa_id=1,
                clube_visitante_id=1,  # Mesmo clube
                rodada=10
            )
        
        assert "diferentes" in str(exc_info.value).lower()
    
    def test_rodada_invalida(self):
        """Testa rejeição de rodada inválida."""
        with pytest.raises(ValidationError):
            PartidaModel(
                partida_id=1,
                clube_casa_id=1,
                clube_visitante_id=2,
                rodada=50  # Campeonato tem 38 rodadas
            )


class TestMercadoStatusModel:
    """Testes para MercadoStatusModel."""
    
    def test_mercado_aberto(self):
        """Testa identificação de mercado aberto."""
        mercado = MercadoStatusModel(
            rodada_atual=10,
            status_mercado=1
        )
        
        assert mercado.mercado_aberto is True
        assert mercado.mercado_fechado is False
    
    def test_mercado_fechado(self):
        """Testa identificação de mercado fechado."""
        mercado = MercadoStatusModel(
            rodada_atual=10,
            status_mercado=2
        )
        
        assert mercado.mercado_aberto is False
        assert mercado.mercado_fechado is True


class TestPredicaoModel:
    """Testes para PredicaoModel."""
    
    def test_predicao_valida(self):
        """Testa criação de predição válida."""
        predicao = PredicaoModel(
            atleta_id=1,
            predicao=12.5,
            predicao_std=2.5,
            confianca=0.85
        )
        
        assert predicao.predicao == 12.5
        assert predicao.confianca == 0.85
    
    def test_predicao_fora_range(self):
        """Testa rejeição de predição absurda."""
        with pytest.raises(ValidationError):
            PredicaoModel(
                atleta_id=1,
                predicao=100.0  # Muito alto
            )
    
    def test_confianca_invalida(self):
        """Testa rejeição de confiança fora de 0-1."""
        with pytest.raises(ValidationError):
            PredicaoModel(
                atleta_id=1,
                predicao=10.0,
                confianca=1.5  # Maior que 1
            )


class TestEscalacaoModel:
    """Testes para EscalacaoModel."""
    
    def test_escalacao_valida(self):
        """Testa criação de escalação válida."""
        escalacao = EscalacaoModel(
            atletas=list(range(1, 13)),  # 12 atletas únicos
            formacao="4-3-3",
            patrimonio_usado=95.0,
            patrimonio_disponivel=100.0,
            pontos_preditos=85.5
        )
        
        assert len(escalacao.atletas) == 12
        assert escalacao.formacao == "4-3-3"
    
    def test_atletas_insuficientes(self):
        """Testa rejeição de escalação com menos de 12 atletas."""
        with pytest.raises(ValidationError):
            EscalacaoModel(
                atletas=[1, 2, 3, 4, 5],  # Apenas 5
                formacao="4-3-3",
                patrimonio_usado=50.0,
                patrimonio_disponivel=100.0,
                pontos_preditos=40.0
            )
    
    def test_atletas_duplicados(self):
        """Testa rejeição de atletas duplicados."""
        with pytest.raises(ValidationError) as exc_info:
            EscalacaoModel(
                atletas=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],  # 11 duplicado
                formacao="4-3-3",
                patrimonio_usado=95.0,
                patrimonio_disponivel=100.0,
                pontos_preditos=85.0
            )
        
        assert "duplicados" in str(exc_info.value).lower()
    
    def test_formacao_invalida(self):
        """Testa rejeição de formação inválida."""
        with pytest.raises(ValidationError):
            EscalacaoModel(
                atletas=list(range(1, 13)),
                formacao="4-4-4",  # Formato inválido
                patrimonio_usado=95.0,
                patrimonio_disponivel=100.0,
                pontos_preditos=85.0
            )
    
    def test_patrimonio_excedido(self):
        """Testa rejeição quando patrimônio usado excede disponível."""
        with pytest.raises(ValidationError) as exc_info:
            EscalacaoModel(
                atletas=list(range(1, 13)),
                formacao="4-3-3",
                patrimonio_usado=150.0,  # Excede disponível
                patrimonio_disponivel=100.0,
                pontos_preditos=85.0
            )
        
        assert "excede" in str(exc_info.value).lower()


class TestValidacaoBatch:
    """Testes para validação em lote."""
    
    def test_todos_validos(self):
        """Testa validação de lista com todos válidos."""
        atletas = [
            {'atleta_id': 1, 'apelido': 'A', 'clube_id': 1, 'posicao_id': 5, 'preco': 10.0},
            {'atleta_id': 2, 'apelido': 'B', 'clube_id': 1, 'posicao_id': 5, 'preco': 15.0},
            {'atleta_id': 3, 'apelido': 'C', 'clube_id': 1, 'posicao_id': 5, 'preco': 20.0},
        ]
        
        validados = validar_atletas_batch(atletas)
        assert len(validados) == 3
    
    def test_alguns_invalidos(self):
        """Testa que inválidos são ignorados."""
        atletas = [
            {'atleta_id': 1, 'apelido': 'A', 'clube_id': 1, 'posicao_id': 5, 'preco': 10.0},
            {'atleta_id': 2, 'apelido': 'B', 'clube_id': 1, 'posicao_id': 5, 'preco': -5.0},  # Inválido
            {'atleta_id': 3, 'apelido': 'C', 'clube_id': 1, 'posicao_id': 5, 'preco': 20.0},
        ]
        
        validados = validar_atletas_batch(atletas)
        assert len(validados) == 2  # Apenas os válidos
