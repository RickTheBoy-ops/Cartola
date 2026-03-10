from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import optimizer_routes

app = FastAPI(
    title="Cartola FC Optimizer API",
    description="API para testes e integrações de otimizações do Cartola FC.",
    version="2.2.0"
)

# CORS Middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API do Cartola FC Optimizer!"}

# Register routers
app.include_router(optimizer_routes.router, prefix="/api/v1", tags=["Optimizer"])
