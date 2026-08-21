#!/usr/bin/env bash
# ==============================================================================
# Mogra Movie Recommender Agent — Automated One-Click Installation & Setup Script
# ==============================================================================
# This script sets up python virtual environment, installs dependencies,
# launches Docker databases (Neo4j & Qdrant), pulls Ollama models, and runs initial dataset ingestion.

set -e

# ANSI Color Code Utilities for Output Formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}"
echo "=============================================================================="
echo "         🎬 MOGRA MOVIE RECOMMENDER AGENT — AUTOMATED INSTALLER 🚀           "
echo "=============================================================================="
echo -e "${NC}"

# Step 1: Check System Prerequisites
echo -e "${BOLD}[1/6] Checking System Prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.10+ and re-run.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  ✅ Python ${PYTHON_VERSION} detected."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH. Please install Docker and re-run.${NC}"
    exit 1
fi
echo -e "  ✅ Docker detected."

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama CLI is not found in PATH. Please ensure Ollama is installed (https://ollama.com).${NC}"
else
    echo -e "  ✅ Ollama CLI detected."
fi

# Step 2: Virtual Environment Setup
echo -e "\n${BOLD}[2/6] Setting Up Python Virtual Environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ✅ Created virtual environment in ./venv"
else
    echo -e "  ✅ Existing virtual environment ./venv found."
fi

source venv/bin/activate
pip install --upgrade pip -q
echo -e "  ✅ Pip upgraded."

echo -e "  📦 Installing Python requirements from requirements.txt..."
pip install -r requirements.txt -q
echo -e "  ✅ All dependencies installed successfully."

# Step 3: Launch Database Services via Docker
echo -e "\n${BOLD}[3/6] Starting Database Containers (Neo4j & Qdrant)...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi
echo -e "  ✅ Docker containers started."

# Step 4: Wait for Database Health Check
echo -e "\n${BOLD}[4/6] Verifying Database Connectivity...${NC}"
echo -n "  ⏳ Waiting for Qdrant Vector Store (http://localhost:6333)... "
until curl -s http://localhost:6333/healthz &> /dev/null; do
    sleep 2
done
echo -e "${GREEN}Connected!${NC}"

echo -n "  ⏳ Waiting for Neo4j Graph DB (http://localhost:7474)... "
until curl -s http://localhost:7474 &> /dev/null; do
    sleep 2
done
echo -e "${GREEN}Connected!${NC}"

# Step 5: Pull Ollama Models
echo -e "\n${BOLD}[5/6] Ensuring Local Ollama Neural Models are Available...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "  🧠 Pulling Qwen 2.5 3B LLM model ('qwen2.5:3b')..."
    ollama pull qwen2.5:3b || echo -e "${YELLOW}⚠️ Could not pull qwen2.5:3b (Ensure Ollama daemon 'ollama serve' is running).${NC}"

    echo -e "  📐 Pulling Nomic Dense Vector Embeddings model ('nomic-embed-text')..."
    ollama pull nomic-embed-text || echo -e "${YELLOW}⚠️ Could not pull nomic-embed-text (Ensure Ollama daemon 'ollama serve' is running).${NC}"
else
    echo -e "${YELLOW}⚠️ Skipping model pull (Ollama CLI not present).${NC}"
fi

# Step 6: Initialize Schemas & Data Ingestion
echo -e "\n${BOLD}[6/6] Initializing Schemas & Running Initial Dataset Ingestion...${NC}"
PYTHONPATH=. ./venv/bin/python3 -c "
from db.neo4j_client import neo4j_client
from db.qdrant_client import qdrant_wrapper
from db.neo4j_schema import initialize_neo4j_schema
from db.qdrant_schema import initialize_qdrant_schema
initialize_neo4j_schema(neo4j_client)
initialize_qdrant_schema(qdrant_wrapper)
print('  ✅ Database constraints & collection schemas initialized.')
" || echo -e "${YELLOW}⚠️ Schema setup warning.${NC}"

if [ -f "scripts/ingest_synopsis_enriched_movies.py" ]; then
    echo -e "  🍿 Running initial movie graph and vector dataset ingestion..."
    PYTHONPATH=. ./venv/bin/python3 scripts/ingest_synopsis_enriched_movies.py || echo -e "${YELLOW}⚠️ Ingestion script skipped or completed with warnings.${NC}"
fi

echo -e "\n${BOLD}${GREEN}"
echo "=============================================================================="
echo "         🎉 INSTALLATION & ENVIRONMENT SETUP COMPLETE! 🍿                    "
echo "=============================================================================="
echo -e "${NC}"
echo -e "${BOLD}To launch the applications, run:${NC}"
echo -e "  ${CYAN}1. Interactive Terminal Chat UI:${NC}"
echo -e "     ${BOLD}PYTHONPATH=. ./venv/bin/python3 tui_app.py${NC}"
echo -e ""
echo -e "  ${CYAN}2. RLHF Human Feedback Evaluation TUI:${NC}"
echo -e "     ${BOLD}PYTHONPATH=. ./venv/bin/python3 feedback_tui_app.py${NC}"
echo -e ""
echo -e "  ${CYAN}3. Production REST API Server & Swagger Documentation:${NC}"
echo -e "     ${BOLD}PYTHONPATH=. ./venv/bin/python3 run_api_server.py --port 8000${NC}"
echo -e "     👉 Interactive Swagger UI: ${BOLD}http://localhost:8000/docs${NC}"
echo -e ""
