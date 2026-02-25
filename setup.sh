#!/usr/bin/env bash
# ============================================================
#  HealthAccess Morocco — Local Bootstrap Script
#  Run this once after cloning the repo.
#  Usage: bash setup.sh
# ============================================================

set -e

echo "🏥 Morocco Healthcare Optimization — Project Setup"
echo "==================================================="

# 1. Python version check
REQUIRED="3.10"
PYTHON=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON (required: $REQUIRED+)"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment..."
  python3.10 -m venv .venv
else
  echo "✓ Virtual environment already exists"
fi

# 3. Activate
source .venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip -q

# 5. Install dependencies
echo "→ Installing dependencies..."
pip install -r requirements.txt -q

# 6. Copy .env if not present
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "✓ Created .env from .env.example — add your API keys there"
else
  echo "✓ .env already exists"
fi

# 7. Run tests to verify setup
echo "→ Running test suite..."
pytest tests/ -v --tb=short

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/fetch_osm_facilities.py    # Phase 1"
echo "  streamlit run webapp/streamlit_app.py      # Phase 4"
