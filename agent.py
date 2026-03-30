#!/usr/bin/env python3
"""
GitHub Repository AI Agent with LangGraph
Analisa repositórios GitHub, propõe melhorias e cria PRs automaticamente
Monitora atualizações e aplica melhorias continuamente
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent
import base64
import hashlib

# Core dependencies
import requests  # type: ignore
from github import Github, GithubException  # type: ignore
from openai import OpenAI  # type: ignore

# LangGraph dependencies
from langgraph.graph import StateGraph, END  # type: ignore
from langgraph.prebuilt import ToolNode  # type: ignore
from langchain_core.messages import HumanMessage, AIMessage  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_core.tools import tool  # type: ignore

# Web framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request  # type: ignore
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PERPLEXITY_API_KEY = str(os.getenv("PERPLEXITY_API_KEY", ""))
GITHUB_TOKEN = str(os.getenv("GITHUB_TOKEN", ""))
WEBHOOK_SECRET = str(os.getenv("WEBHOOK_SECRET", "default_secret"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

if not PERPLEXITY_API_KEY or not GITHUB_TOKEN:
    raise ValueError("PERPLEXITY_API_KEY e GITHUB_TOKEN devem ser definidos nas variáveis de ambiente. Você pode configurá-las no arquivo .env")

# Initialize clients
perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
github_client = Github(GITHUB_TOKEN)
llm = ChatOpenAI(
    model="llama-3.1-sonar-large-128k-online",
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai",
    temperature=LLM_TEMPERATURE
)

# State definitions
class AgentState(TypedDict):
    messages: Annotated[List[dict], "Mensagens da conversação"]
    repository_url: str
    repository_data: dict
    analysis_result: dict
    improvement_plan: dict
    user_approval: bool
    pr_created: bool
    monitoring_active: bool
    last_commit_sha: str
    error_messages: List[str]

@dataclass
class RepositoryAnalysis:
    """Estrutura para análise do repositório"""
    code_quality_score: float
    security_issues: List[str]
    performance_issues: List[str]
    documentation_issues: List[str]
    best_practices_violations: List[str]
    suggested_improvements: List[dict]
    priority_level: str

@dataclass
class ImprovementPlan:
    """Plano de melhorias estruturado"""
    high_priority: List[dict]
    medium_priority: List[dict]
    low_priority: List[dict]
    estimated_time: str
    files_to_modify: List[str]
    new_files_to_create: List[str]

class GitHubRepositoryManager:
    """Gerenciador de operações do GitHub"""
    
    def __init__(self, token: str):
        self.github = Github(token)
        self.token = token
    
    def get_repository_info(self, repo_url: str) -> dict:
        """Extrai informações detalhadas do repositório"""
        try:
            # Parse repository URL
            if "github.com" in repo_url:
                repo_path = repo_url.split("github.com/")[-1].replace(".git", "")
            else:
                repo_path = repo_url
            
            repo = self.github.get_repo(repo_path)
            
            # Get repository structure
            contents = self._get_repository_contents(repo)
            
            # Get recent commits
            commits = list(repo.get_commits()[:10])
            
            # Get languages
            languages = repo.get_languages()
            
            # Get issues and PRs
            open_issues = list(repo.get_issues(state='open')[:20])
            open_prs = list(repo.get_pulls(state='open')[:10])
            
            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "language": repo.language,
                "languages": languages,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "size": repo.size,
                "default_branch": repo.default_branch,
                "contents": contents,
                "recent_commits": [
                    {
                        "sha": commit.sha,
                        "message": commit.commit.message,
                        "author": commit.commit.author.name,
                        "date": commit.commit.author.date.isoformat()
                    }
                    for commit in commits
                ],
                "open_issues_count": len(open_issues),
                "open_prs_count": len(open_prs),
                "topics": repo.get_topics(),
                "has_readme": self._check_file_exists(repo, "README.md"),
                "has_license": repo.license is not None,
                "has_gitignore": self._check_file_exists(repo, ".gitignore"),
                "has_ci": self._check_ci_files(repo),
                "last_updated": repo.updated_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Erro ao obter informações do repositório: {e}")
            raise
    
    def _get_repository_contents(self, repo, path="", max_files=100) -> dict:
        """Obtém o conteúdo do repositório de forma recursiva"""
        contents = {"files": [], "directories": []}
        file_count = 0
        
        try:
            items = repo.get_contents(path)
            if not isinstance(items, list):
                items = [items]
            
            for item in items:
                if file_count >= max_files:
                    break
                
                if item.type == "dir":
                    contents["directories"].append({
                        "name": item.name,
                        "path": item.path
                    })
                else:
                    file_info = {
                        "name": item.name,
                        "path": item.path,
                        "size": item.size,
                        "type": item.name.split('.')[-1] if '.' in item.name else 'unknown'
                    }
                    
                    # Get file content for analysis (only for small files)
                    if item.size < 50000:  # 50KB limit
                        try:
                            content = base64.b64decode(item.content).decode('utf-8')
                            file_info["content"] = content
                        except:
                            file_info["content"] = "Binary file or encoding error"
                    
                    contents["files"].append(file_info)
                    file_count += 1
                    
        except Exception as e:
            logger.warning(f"Erro ao acessar conteúdo em {path}: {e}")
        
        return contents
    
    def _check_file_exists(self, repo, filename: str) -> bool:
        """Verifica se um arquivo existe no repositório"""
        try:
            repo.get_contents(filename)
            return True
        except:
            return False
    
    def _check_ci_files(self, repo) -> bool:
        """Verifica se existem arquivos de CI/CD"""
        ci_paths = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            ".travis.yml",
            "circle.yml"
        ]
        
        for path in ci_paths:
            if self._check_file_exists(repo, path):
                return True
        return False
    
    def create_pull_request(self, repo_path: str, improvements: dict) -> str:
        """Cria uma pull request com as melhorias"""
        try:
            repo = self.github.get_repo(repo_path)
            
            # Create a new branch
            base_branch = repo.default_branch
            new_branch = f"ai-improvements-{int(time.time())}"
            
            # Get base branch reference
            base_ref = repo.get_git_ref(f"heads/{base_branch}")
            repo.create_git_ref(
                ref=f"refs/heads/{new_branch}",
                sha=base_ref.object.sha
            )
            
            # Apply improvements to files
            for improvement in improvements.get("file_changes", []):
                file_path = improvement["file_path"]
                new_content = improvement["new_content"]
                commit_message = f"AI Improvement: {improvement['description']}"
                
                try:
                    # Try to get existing file
                    file = repo.get_contents(file_path, ref=new_branch)
                    repo.update_file(
                        path=file_path,
                        message=commit_message,
                        content=new_content,
                        sha=file.sha,
                        branch=new_branch
                    )
                except:
                    # Create new file if it doesn't exist
                    repo.create_file(
                        path=file_path,
                        message=commit_message,
                        content=new_content,
                        branch=new_branch
                    )
            
            # Create pull request
            pr_title = "🤖 AI-Generated Repository Improvements"
            pr_body = self._generate_pr_description(improvements)
            
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=new_branch,
                base=base_branch
            )
            
            return pr.html_url
            
        except Exception as e:
            logger.error(f"Erro ao criar pull request: {e}")
            raise
    
    def _generate_pr_description(self, improvements: dict) -> str:
        """Gera descrição detalhada da PR"""
        description = """# 🤖 AI-Generated Repository Improvements

This pull request contains automated improvements suggested by our AI agent.

## 📊 Analysis Summary
"""
        
        if "analysis_summary" in improvements:
            description += f"- **Code Quality Score**: {improvements['analysis_summary'].get('code_quality_score', 'N/A')}\n"
            description += f"- **Priority Level**: {improvements['analysis_summary'].get('priority_level', 'Medium')}\n\n"
        
        description += "## 🔧 Changes Made\n\n"
        
        for change in improvements.get("file_changes", []):
            description += f"- **{change['file_path']}**: {change['description']}\n"
        
        description += "\n## ✅ Benefits\n\n"
        for benefit in improvements.get("benefits", []):
            description += f"- {benefit}\n"
        
        description += "\n---\n*This PR was automatically generated by AI. Please review carefully before merging.*"
        
        return description

class CodeAnalyzer:
    """Analisador de código usando Perplexity"""
    
    def __init__(self, client):
        self.client = client
    
    def analyze_repository(self, repo_data: dict) -> RepositoryAnalysis:
        """Análise completa do repositório"""
        try:
            analysis_prompt = self._create_analysis_prompt(repo_data)
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um especialista em análise de código e arquitetura de software.
                        Analise o repositório fornecido e retorne um JSON estruturado com:
                        - code_quality_score (0-100)
                        - security_issues (lista)
                        - performance_issues (lista)
                        - documentation_issues (lista)
                        - best_practices_violations (lista)
                        - suggested_improvements (lista de objetos com title, description, priority, files_affected)
                        - priority_level (high/medium/low)
                        
                        Seja específico e prático nas sugestões."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            analysis_json = json.loads(response.choices[0].message.content)
            
            return RepositoryAnalysis(
                code_quality_score=analysis_json.get("code_quality_score", 70.0),
                security_issues=analysis_json.get("security_issues", []),
                performance_issues=analysis_json.get("performance_issues", []),
                documentation_issues=analysis_json.get("documentation_issues", []),
                best_practices_violations=analysis_json.get("best_practices_violations", []),
                suggested_improvements=analysis_json.get("suggested_improvements", []),
                priority_level=analysis_json.get("priority_level", "medium")
            )
            
        except Exception as e:
            logger.error(f"Erro na análise do código: {e}")
            # Return default analysis on error
            return RepositoryAnalysis(
                code_quality_score=50.0,
                security_issues=["Erro na análise de segurança"],
                performance_issues=["Erro na análise de performance"],
                documentation_issues=["Erro na análise de documentação"],
                best_practices_violations=["Erro na análise de boas práticas"],
                suggested_improvements=[{
                    "title": "Análise detalhada necessária",
                    "description": "Execute uma análise manual devido a erro na análise automática",
                    "priority": "high",
                    "files_affected": []
                }],
                priority_level="high"
            )
    
    def _create_analysis_prompt(self, repo_data: dict) -> str:
        """Cria prompt detalhado para análise"""
        prompt = f"""
Analise o seguinte repositório GitHub:

**Informações Gerais:**
- Nome: {repo_data.get('name', 'N/A')}
- Linguagem Principal: {repo_data.get('language', 'N/A')}
- Linguagens: {repo_data.get('languages', {})}
- Descrição: {repo_data.get('description', 'N/A')}
- Tópicos: {repo_data.get('topics', [])}

**Estrutura do Projeto:**
- Tem README: {repo_data.get('has_readme', False)}
- Tem LICENSE: {repo_data.get('has_license', False)}
- Tem .gitignore: {repo_data.get('has_gitignore', False)}
- Tem CI/CD: {repo_data.get('has_ci', False)}

**Arquivos Principais:**
"""
        
        # Add file contents for analysis
        for file_info in repo_data.get('contents', {}).get('files', [])[:10]:  # Limit to 10 files
            if file_info.get('content') and file_info['size'] < 10000:
                prompt += f"\n--- {file_info['path']} ---\n{file_info['content'][:1000]}...\n"  # type: ignore
        
        prompt += """  # type: ignore

Forneça uma análise detalhada focando em:
1. Qualidade do código (estrutura, legibilidade, manutenibilidade)
2. Segurança (vulnerabilidades, exposição de dados sensíveis)
3. Performance (otimizações possíveis, gargalos)
4. Documentação (completude, clareza, exemplos)
5. Boas práticas (padrões de projeto, arquitetura, testes)

Retorne APENAS um JSON válido com a estrutura solicitada.
"""
        
        return prompt
    
    def generate_improvements(self, analysis: RepositoryAnalysis, repo_data: dict) -> dict:
        """Gera melhorias práticas baseadas na análise"""
        try:
            improvement_prompt = f"""
Com base na análise do repositório, gere melhorias práticas:

**Análise:**
- Score de Qualidade: {analysis.code_quality_score}
- Problemas de Segurança: {analysis.security_issues}
- Problemas de Performance: {analysis.performance_issues}
- Problemas de Documentação: {analysis.documentation_issues}
- Violações de Boas Práticas: {analysis.best_practices_violations}

**Sugestões de Melhoria:**
{json.dumps(analysis.suggested_improvements, indent=2)}

Gere um JSON com:
- file_changes: lista de objetos com file_path, new_content, description
- benefits: lista de benefícios das mudanças
- estimated_time: tempo estimado para implementação
- priority_order: ordem de implementação

Foque em mudanças práticas e implementáveis. Gere conteúdo real de arquivos.
"""
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um desenvolvedor experiente que gera melhorias práticas e implementáveis para repositórios."
                    },
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Erro ao gerar melhorias: {e}")
            return {
                "file_changes": [],
                "benefits": ["Melhorias não puderam ser geradas automaticamente"],
                "estimated_time": "N/A",
                "priority_order": []
            }

# LangGraph Tools
@tool
def analyze_repository_tool(repository_url: str) -> dict:
    """Ferramenta para analisar repositório GitHub"""
    try:
        github_manager = GitHubRepositoryManager(GITHUB_TOKEN)
        repo_data = github_manager.get_repository_info(repository_url)
        
        analyzer = CodeAnalyzer(perplexity_client)
        analysis = analyzer.analyze_repository(repo_data)
        
        return {
            "repository_data": repo_data,
            "analysis": {
                "code_quality_score": analysis.code_quality_score,
                "security_issues": analysis.security_issues,
                "performance_issues": analysis.performance_issues,
                "documentation_issues": analysis.documentation_issues,
                "best_practices_violations": analysis.best_practices_violations,
                "suggested_improvements": analysis.suggested_improvements,
                "priority_level": analysis.priority_level
            }
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def generate_improvement_plan_tool(analysis_result: dict) -> dict:
    """Ferramenta para gerar plano de melhorias"""
    try:
        analyzer = CodeAnalyzer(perplexity_client)
        repo_data = analysis_result["repository_data"]
        analysis = RepositoryAnalysis(**analysis_result["analysis"])  # type: ignore
        
        improvements = analyzer.generate_improvements(analysis, repo_data)
        return improvements
    except Exception as e:
        return {"error": str(e)}

@tool
def create_pull_request_tool(repo_url: str, improvements: dict) -> dict:
    """Ferramenta para criar pull request"""
    try:
        github_manager = GitHubRepositoryManager(GITHUB_TOKEN)
        
        # Extract repo path from URL
        if "github.com" in repo_url:
            repo_path = repo_url.split("github.com/")[-1].replace(".git", "")
        else:
            repo_path = repo_url
        
        pr_url = github_manager.create_pull_request(repo_path, improvements)
        return {"pr_url": pr_url, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

# LangGraph Node Functions
# (... resto do arquivo permanece igual ...)
"""
Para colocar o agente em produção com este arquivo `agent.py`:

1. Instalar dependências:
pip install -r requirements.txt

2. Configurar variáveis de ambiente (ou arquivo .env):
export PERPLEXITY_API_KEY="sua_api_key_da_perplexity"
export GITHUB_TOKEN="seu_github_token"
export WEBHOOK_SECRET="seu_webhook_secret"
export LLM_TEMPERATURE="0.1"

3. Executar em modo servidor:
python agent.py --mode server --host 0.0.0.0 --port 8000

4. Configurar webhook no GitHub:
- URL: https://seu-dominio.com/webhook/github
- Content type: application/json
- Events: push, pull_request

5. Para Docker (exemplo mínimo):
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY agent.py .
CMD ["python", "agent.py", "--mode", "server"]

6. Para uso direto (CLI):
python agent.py --mode cli
python agent.py --mode cli --repo https://github.com/user/repo

Principais endpoints expostos pelo FastAPI:
- POST /analyze-repository - Inicia análise
- GET /session/{id}/status - Status da sessão
- POST /session/approve - Aprova melhorias
- DELETE /session/{id} - Para monitoramento
- GET /sessions - Lista sessões ativas
- GET /health - Health check
- POST /webhook/github - Webhook do GitHub
"""
