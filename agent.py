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
def repository_analysis_node(state: AgentState) -> AgentState:
    """Nó para análise do repositório"""
    try:
        result = analyze_repository_tool.invoke({"repository_url": state["repository_url"]})
        
        if "error" in result:
            state["error_messages"].append(f"Erro na análise: {result['error']}")
        else:
            state["repository_data"] = result["repository_data"]
            state["analysis_result"] = result["analysis"]
            
            # Add analysis message
            state["messages"].append({
                "role": "assistant",
                "content": f"✅ Análise concluída!\n\n"
                          f"**Score de Qualidade:** {result['analysis']['code_quality_score']}/100\n"
                          f"**Problemas encontrados:**\n"
                          f"- Segurança: {len(result['analysis']['security_issues'])} problemas\n"
                          f"- Performance: {len(result['analysis']['performance_issues'])} problemas\n"
                          f"- Documentação: {len(result['analysis']['documentation_issues'])} problemas\n"
                          f"- Boas Práticas: {len(result['analysis']['best_practices_violations'])} violações\n\n"
                          f"**Prioridade:** {result['analysis']['priority_level'].upper()}\n"
                          f"**Melhorias sugeridas:** {len(result['analysis']['suggested_improvements'])}"
            })
        
        return state
    except Exception as e:
        state["error_messages"].append(f"Erro no nó de análise: {str(e)}")
        return state

def improvement_planning_node(state: AgentState) -> AgentState:
    """Nó para planejamento de melhorias"""
    try:
        if not state["analysis_result"]:
            state["error_messages"].append("Análise necessária antes do planejamento")
            return state
        
        result = generate_improvement_plan_tool.invoke({"analysis_result": {
            "repository_data": state["repository_data"],
            "analysis": state["analysis_result"]
        }})
        
        if "error" in result:
            state["error_messages"].append(f"Erro no planejamento: {result['error']}")
        else:
            state["improvement_plan"] = result
            
            # Add planning message
            file_changes = len(result.get("file_changes", []))
            benefits = len(result.get("benefits", []))
            
            state["messages"].append({
                "role": "assistant",
                "content": f"📋 Plano de melhorias criado!\n\n"
                          f"**Alterações planejadas:** {file_changes} arquivos\n"
                          f"**Benefícios esperados:** {benefits} melhorias\n"
                          f"**Tempo estimado:** {result.get('estimated_time', 'N/A')}\n\n"
                          f"**Você aprova a implementação das melhorias?**\n"
                          f"Digite 'sim' para aprovar ou 'não' para cancelar."
            })
        
        return state
    except Exception as e:
        state["error_messages"].append(f"Erro no nó de planejamento: {str(e)}")
        return state

def implementation_node(state: AgentState) -> AgentState:
    """Nó para implementação das melhorias"""
    try:
        if not state["user_approval"]:
            state["messages"].append({
                "role": "assistant",
                "content": "❌ Implementação cancelada pelo usuário."
            })
            return state
        
        if not state["improvement_plan"]:
            state["error_messages"].append("Plano de melhorias necessário para implementação")
            return state
        
        result = create_pull_request_tool.invoke({
            "repo_url": state["repository_url"],
            "improvements": state["improvement_plan"]
        })
        
        if result["success"]:
            state["pr_created"] = True
            state["messages"].append({
                "role": "assistant",
                "content": f"🚀 Pull Request criada com sucesso!\n\n"
                          f"**URL da PR:** {result['pr_url']}\n\n"
                          f"As melhorias foram implementadas e estão aguardando sua revisão.\n"
                          f"O monitoramento contínuo do repositório foi ativado."
            })
            
            # Start monitoring
            state["monitoring_active"] = True
            state["last_commit_sha"] = state["repository_data"].get("recent_commits", [{}])[0].get("sha", "")
            
        else:
            state["error_messages"].append(f"Erro na criação da PR: {result.get('error', 'Erro desconhecido')}")
        
        return state
    except Exception as e:
        state["error_messages"].append(f"Erro no nó de implementação: {str(e)}")
        return state

def monitoring_node(state: AgentState) -> AgentState:
    """Nó para monitoramento contínuo"""
    try:
        if not state["monitoring_active"]:
            return state
        
        # Check for new commits
        github_manager = GitHubRepositoryManager(GITHUB_TOKEN)
        current_repo_data = github_manager.get_repository_info(state["repository_url"])
        
        current_commit_sha = current_repo_data.get("recent_commits", [{}])[0].get("sha", "")
        
        if current_commit_sha != state["last_commit_sha"] and state["last_commit_sha"]:
            state["messages"].append({
                "role": "assistant",
                "content": "🔍 Novas alterações detectadas no repositório!\n"
                          "Iniciando nova análise para identificar possíveis melhorias..."
            })
            
            # Trigger new analysis cycle
            state["repository_data"] = current_repo_data
            state["analysis_result"] = {}
            state["improvement_plan"] = {}
            state["user_approval"] = False
            state["pr_created"] = False
            state["last_commit_sha"] = current_commit_sha
            
            # Run new analysis
            return repository_analysis_node(state)
        
        return state
    except Exception as e:
        state["error_messages"].append(f"Erro no monitoramento: {str(e)}")
        return state

# Create LangGraph workflow
def create_workflow() -> StateGraph:
    """Cria o workflow do LangGraph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", repository_analysis_node)
    workflow.add_node("plan", improvement_planning_node)
    workflow.add_node("implement", implementation_node)
    workflow.add_node("monitor", monitoring_node)
    
    # Define edges
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "implement")
    workflow.add_edge("implement", "monitor")
    workflow.add_edge("monitor", "analyze")  # Loop for continuous monitoring
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    return workflow.compile()

# FastAPI Application
app = FastAPI(title="GitHub AI Agent", description="Agente de IA para análise e melhoria de repositórios GitHub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Global workflow instance
workflow = create_workflow()

# Active sessions storage management
import json

SESSION_FILE = ROOT_DIR / "data" / "agent_sessions.json"

def load_sessions() -> Dict[str, Any]:
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar sessões: {e}")
    return {}

def save_sessions():
    try:
        data_to_save = {}
        for k, v in active_sessions.items():
            state_dict = dict(v)
            if "messages" in state_dict:
                state_dict["messages"] = []  # Clear messages to allow standard JSON serialization
            data_to_save[k] = state_dict
            
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar sessões: {e}")

active_sessions: Dict[str, Any] = load_sessions()

# Pydantic models
class RepositoryAnalysisRequest(BaseModel):
    repository_url: str

class UserApprovalRequest(BaseModel):
    session_id: str
    approved: bool

class MonitoringStatusRequest(BaseModel):
    session_id: str

@app.post("/analyze-repository")
async def analyze_repository_endpoint(
    request: RepositoryAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Endpoint para iniciar análise de repositório"""
    try:
        session_id = hashlib.md5(f"{request.repository_url}{time.time()}".encode()).hexdigest()
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            repository_url=request.repository_url,
            repository_data={},
            analysis_result={},
            improvement_plan={},
            user_approval=False,
            pr_created=False,
            monitoring_active=False,
            last_commit_sha="",
            error_messages=[]
        )
        
        # Store session
        active_sessions[session_id] = initial_state
        save_sessions()
        
        # Start analysis in background
        background_tasks.add_task(run_analysis_workflow, session_id)
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Análise iniciada. Use o session_id para acompanhar o progresso."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Endpoint para verificar status da sessão"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    state = active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "messages": state["messages"],
        "analysis_complete": bool(state["analysis_result"]),
        "plan_ready": bool(state["improvement_plan"]),
        "pr_created": state["pr_created"],
        "monitoring_active": state["monitoring_active"],
        "errors": state["error_messages"]
    }

@app.post("/session/approve")
async def approve_improvements(request: UserApprovalRequest, background_tasks: BackgroundTasks):
    """Endpoint para aprovar implementação de melhorias"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    state = active_sessions[request.session_id]
    state["user_approval"] = request.approved
    
    if request.approved:
        # Continue workflow with implementation
        background_tasks.add_task(run_implementation_workflow, request.session_id)
        message = "Melhorias aprovadas. Implementação iniciada."
    else:
        message = "Melhorias rejeitadas pelo usuário."
    
    return {
        "session_id": request.session_id,
        "status": "approved" if request.approved else "rejected",
        "message": message
    }

import hmac
import hashlib

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Webhook para receber notificações do GitHub"""
    try:
        body = await request.body()
        signature_header = request.headers.get("X-Hub-Signature-256")
        
        if not signature_header:
            raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
            
        hash_object = hmac.new(WEBHOOK_SECRET.encode('utf-8'), msg=body, digestmod=hashlib.sha256)
        expected_signature = "sha256=" + hash_object.hexdigest()
        
        if not hmac.compare_digest(expected_signature, signature_header):
            raise HTTPException(status_code=403, detail="Invalid signature")

        payload = await request.json()
        
        if "repository" not in payload:
            raise HTTPException(status_code=400, detail="Invalid webhook payload")
        
        repo_url = payload["repository"]["html_url"]
        
        # Find active monitoring sessions for this repository
        for session_id, state in active_sessions.items():
            if (state.get("repository_url") == repo_url and 
                state.get("monitoring_active")):
                
                # Trigger monitoring check
                background_tasks.add_task(run_monitoring_check, session_id)
        
        return {"status": "webhook processed"}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erro no processamento do webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal Error Webhook")

@app.delete("/session/{session_id}")
async def stop_monitoring(session_id: str):
    """Endpoint para parar monitoramento de uma sessão"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    state = active_sessions[session_id]
    state["monitoring_active"] = False
    
    return {
        "session_id": session_id,
        "status": "monitoring stopped",
        "message": "Monitoramento do repositório foi interrompido."
    }

@app.get("/sessions")
async def list_active_sessions():
    """Endpoint para listar sessões ativas"""
    sessions = []
    for session_id, state in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "repository_url": state["repository_url"],
            "monitoring_active": state["monitoring_active"],
            "pr_created": state["pr_created"],
            "last_activity": datetime.now().isoformat()
        })
    
    return {"active_sessions": sessions, "total": len(sessions)}

# Background task functions
async def run_analysis_workflow(session_id: str):
    """Executa workflow de análise em background"""
    try:
        if session_id not in active_sessions:
            return
        
        state = active_sessions[session_id]
        
        # Run analysis
        updated_state = workflow.invoke(state)
        active_sessions[session_id] = updated_state
        
    except Exception as e:
        logger.error(f"Erro no workflow de análise: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["error_messages"].append(f"Erro no workflow: {str(e)}")

async def run_implementation_workflow(session_id: str):
    """Executa workflow de implementação em background"""
    try:
        if session_id not in active_sessions:
            return
        
        state = active_sessions[session_id]
        
        # Run implementation node
        updated_state = implementation_node(state)
        active_sessions[session_id] = updated_state
        
    except Exception as e:
        logger.error(f"Erro no workflow de implementação: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["error_messages"].append(f"Erro na implementação: {str(e)}")

async def run_monitoring_check(session_id: str):
    """Executa verificação de monitoramento em background"""
    try:
        if session_id not in active_sessions:
            return
        
        state = active_sessions[session_id]
        
        # Run monitoring node
        updated_state = monitoring_node(state)
        active_sessions[session_id] = updated_state
        
    except Exception as e:
        logger.error(f"Erro no monitoramento: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["error_messages"].append(f"Erro no monitoramento: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "perplexity_available": PERPLEXITY_API_KEY is not None,
        "github_available": GITHUB_TOKEN is not None
    }

# Cleanup task for old sessions
async def cleanup_old_sessions():
    """Remove sessões antigas para evitar vazamento de memória"""
    while True:
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, state in active_sessions.items():
                # Remove sessions older than 24 hours that are not monitoring
                if not state["monitoring_active"]:
                    # Simple time-based cleanup (you might want to add timestamp to state)
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions (keep only last 100 non-monitoring sessions)
            if len(sessions_to_remove) > 100:
                for session_id in sessions_to_remove[:-100]:  # type: ignore
                    del active_sessions[session_id]  # type: ignore
            
            await asyncio.sleep(3600)  # Run every hour
            
        except Exception as e:
            logger.error(f"Erro na limpeza de sessões: {e}")
            await asyncio.sleep(3600)

# CLI Interface for direct usage
class GitHubAIAgentCLI:
    """Interface de linha de comando para o agente"""
    
    def __init__(self):
        self.workflow = create_workflow()
        self.github_manager = GitHubRepositoryManager(GITHUB_TOKEN)
        self.analyzer = CodeAnalyzer(perplexity_client)
    
    def run_interactive(self):
        """Executa modo interativo"""
        print("🤖 GitHub AI Agent - Modo Interativo")
        print("="*50)
        
        while True:
            try:
                repo_url = input("\nDigite a URL do repositório GitHub (ou 'quit' para sair): ").strip()
                
                if repo_url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Até logo!")
                    break
                
                if not repo_url:
                    continue
                
                # Initialize state
                state = AgentState(
                    messages=[],
                    repository_url=repo_url,
                    repository_data={},
                    analysis_result={},
                    improvement_plan={},
                    user_approval=False,
                    pr_created=False,
                    monitoring_active=False,
                    last_commit_sha="",
                    error_messages=[]
                )
                
                print(f"\n🔍 Analisando repositório: {repo_url}")
                
                # Run analysis
                state = repository_analysis_node(state)
                
                if state["error_messages"]:
                    print(f"❌ Erros na análise: {', '.join(state['error_messages'])}")
                    continue
                
                # Display analysis results
                analysis = state["analysis_result"]
                print(f"\n📊 Resultados da Análise:")
                print(f"Score de Qualidade: {analysis['code_quality_score']}/100")
                print(f"Problemas de Segurança: {len(analysis['security_issues'])}")
                print(f"Problemas de Performance: {len(analysis['performance_issues'])}")
                print(f"Problemas de Documentação: {len(analysis['documentation_issues'])}")
                print(f"Violações de Boas Práticas: {len(analysis['best_practices_violations'])}")
                print(f"Melhorias Sugeridas: {len(analysis['suggested_improvements'])}")
                
                # Generate improvement plan
                print(f"\n📋 Gerando plano de melhorias...")
                state = improvement_planning_node(state)
                
                if state["error_messages"]:
                    print(f"❌ Erros no planejamento: {', '.join(state['error_messages'])}")
                    continue
                
                # Display improvement plan
                plan = state["improvement_plan"]
                print(f"\n🔧 Plano de Melhorias:")
                print(f"Arquivos a serem modificados: {len(plan.get('file_changes', []))}")
                print(f"Benefícios esperados: {len(plan.get('benefits', []))}")
                print(f"Tempo estimado: {plan.get('estimated_time', 'N/A')}")
                
                # Ask for approval
                approval = input(f"\n✅ Deseja implementar as melhorias? (s/n): ").lower().strip()
                
                if approval in ['s', 'sim', 'y', 'yes']:
                    state["user_approval"] = True
                    print(f"\n🚀 Implementando melhorias...")
                    
                    # Implement improvements
                    state = implementation_node(state)
                    
                    if state["pr_created"]:
                        print(f"✅ Pull Request criada com sucesso!")
                        
                        # Ask about monitoring
                        monitor = input(f"\n🔍 Deseja ativar monitoramento contínuo? (s/n): ").lower().strip()
                        
                        if monitor in ['s', 'sim', 'y', 'yes']:
                            print(f"📡 Monitoramento ativado. O agente verificará mudanças periodicamente.")
                            print(f"💡 Use o modo servidor (FastAPI) para monitoramento em produção.")
                    else:
                        print(f"❌ Erro ao criar Pull Request: {', '.join(state['error_messages'])}")
                else:
                    print(f"❌ Implementação cancelada pelo usuário.")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Operação cancelada. Até logo!")
                break
            except Exception as e:
                print(f"❌ Erro inesperado: {e}")

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub AI Agent")
    parser.add_argument("--mode", choices=["server", "cli"], default="server",
                       help="Modo de execução: server (FastAPI) ou cli (interativo)")
    parser.add_argument("--host", default="0.0.0.0", help="Host para o servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta para o servidor")
    parser.add_argument("--repo", help="URL do repositório para análise direta")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        cli = GitHubAIAgentCLI()
        
        if args.repo:
            # Direct analysis mode
            print(f"🔍 Analisando repositório: {args.repo}")
            # Add direct analysis logic here
        else:
            # Interactive mode
            cli.run_interactive()
    
    else:
        # Server mode
        print(f"🚀 Iniciando GitHub AI Agent Server...")
        print(f"📡 Servidor rodando em http://{args.host}:{args.port}")
        print(f"📖 Documentação disponível em http://{args.host}:{args.port}/docs")
        print(f"💡 Health check: http://{args.host}:{args.port}/health")
        
        # Start cleanup task
        asyncio.create_task(cleanup_old_sessions())
        
        # Run server
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            log_level="info"
        )

if __name__ == "__main__":
    main()

# Production deployment helper
"""
Para colocar em produção:

1. Instalar dependências:
pip install fastapi uvicorn langgraph langchain-groq PyGithub groq requests

2. Configurar variáveis de ambiente:
export GROQ_API_KEY="sua_groq_api_key"
export GITHUB_TOKEN="seu_github_token"
export WEBHOOK_SECRET="seu_webhook_secret"

3. Executar em modo servidor:
python github_ai_agent.py --mode server --host 0.0.0.0 --port 8000

4. Configurar webhook no GitHub:
- URL: https://seu-dominio.com/webhook/github
- Content type: application/json
- Events: push, pull_request

5. Para Docker:
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY github_ai_agent.py .
CMD ["python", "github_ai_agent.py", "--mode", "server"]

6. Para uso direto (CLI):
python github_ai_agent.py --mode cli
python github_ai_agent.py --mode cli --repo https://github.com/user/repo

APIs disponíveis:
- POST /analyze-repository - Inicia análise
- GET /session/{id}/status - Status da sessão
- POST /session/approve - Aprova melhorias
- DELETE /session/{id} - Para monitoramento
- GET /sessions - Lista sessões ativas
- GET /health - Health check
- POST /webhook/github - Webhook do GitHub
"""