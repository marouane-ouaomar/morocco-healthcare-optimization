# 🏥 Morocco Healthcare Optimization

[![CI](https://github.com/YOUR_USERNAME/morocco-healthcare-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/morocco-healthcare-optimization/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP.streamlit.app)

> **A geospatial healthcare access analysis and optimization platform for Morocco.**  
> Built for data scientists, GIS analysts, and public health professionals.

---

## 📌 What This Project Does

HealthAccess Morocco maps **where healthcare facilities exist**, measures **how accessible they are** to the population, identifies **underserved regions**, and simulates the impact of **adding new facilities or mobile health units**.

| Layer | What it answers |
|---|---|
| 🗺 **Facility Map** | Where are hospitals, clinics, pharmacies, and doctors in Morocco? |
| 📊 **Access Metrics** | What % of the population lives within 5 / 10 / 20 km of a facility? |
| ⚙️ **Optimization** | Where should new facilities be placed to maximize coverage? |
| 🤖 **Triage Bot** | Can a safe, rule-based AI help patients know when to seek emergency care? |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/morocco-healthcare-optimization.git
cd morocco-healthcare-optimization

# 2. Set up Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fetch facility data for Morocco
python scripts/fetch_osm_facilities.py

# 5. Run the Streamlit dashboard
streamlit run webapp/streamlit_app.py
```

---

## 📂 Project Structure

```
healthaccess-morocco/
├── src/
│   ├── data_prep.py          # Facility & population data cleaning
│   ├── access_metrics.py     # KD-tree nearest-neighbor, coverage ratios
│   ├── kmeans_placement.py   # Weighted KMeans optimization
│   ├── or_tools_placement.py # MIP solver (optional, advanced)
│   ├── scenario_simulator.py # Scenario runner & output
│   └── triage_engine.py      # Safe rule-based triage bot
├── scripts/
│   └── fetch_osm_facilities.py  # Overpass API → GeoJSON
├── data/
│   ├── raw/                  # Raw inputs (never committed: *.tiff)
│   └── processed/            # Clean GeoJSON & CSV outputs
├── tests/                    # pytest test suite
├── docs/
│   ├── SAFETY.md             # Triage bot safety documentation
│   └── EVALUATION.md         # Coverage & optimization metrics
├── webapp/
│   └── streamlit_app.py      # Interactive dashboard
├── .github/workflows/ci.yml  # GitHub Actions CI
├── .env.example              # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🗺 Data Sources

| Dataset | Source | License |
|---|---|---|
| Healthcare facilities | [OpenStreetMap](https://www.openstreetmap.org/) via [Overpass API](https://overpass-api.de/) | ODbL |
| Population grid | [WorldPop](https://www.worldpop.org/) (Morocco 100m grid) | CC BY 4.0 |
| Administrative boundaries | [GADM](https://gadm.org/) / OpenStreetMap | See source |

> ⚠️ **Raw population rasters (TIFF) are not committed.** Only aggregated GeoJSON samples are included.

---

## 📈 Key Metrics (Morocco Baseline)

| Metric | Value |
|---|---|
| Facilities mapped | *run pipeline to compute* |
| Population within 5 km of a facility | *run pipeline* |
| Population within 10 km | *run pipeline* |
| Average nearest-facility distance | *run pipeline* |
| Regions with ratio > 10,000 pop/facility | *run pipeline* |

---

## ⚙️ Running the Full Pipeline

```bash
# Step 1 — Fetch & clean facility data
python scripts/fetch_osm_facilities.py
python -c "from src.data_prep import run_pipeline; run_pipeline()"

# Step 2 — Compute access metrics
python -c "from src.access_metrics import run_metrics; run_metrics()"

# Step 3 — Run scenario optimization
python -c "from src.scenario_simulator import run_scenario; run_scenario(new_facilities=5)"

# Step 4 — Launch dashboard
streamlit run webapp/streamlit_app.py

# Run all tests
pytest tests/ -v
```

---

## 🤖 Triage Bot

The triage assistant is a **research demo only**.

- ✅ Detects emergencies: chest pain, breathing difficulty, stroke signs, severe bleeding
- ✅ Outputs only structured JSON advice (no diagnosis)
- ✅ Always recommends professional consultation
- ✅ Local fallback mode — no API key required for demos
- ❌ Not a medical device
- ❌ Not a substitute for clinical care

See [`docs/SAFETY.md`](docs/SAFETY.md) for the full safety specification.

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage targets:
- Data cleaning functions (synthetic fixtures)
- Access metric calculations (known-answer tests)
- Scenario simulator (synthetic population grid)
- Triage emergency detection (100-case synthetic set)

---

## 🌍 Ethics & Data Privacy

This project handles **no personal data**. Specifically:

- ✅ Only aggregated population grids (no individual records)
- ✅ No names, addresses, or identifiers linked to individuals
- ✅ No sensitive medical data stored or transmitted
- ✅ Facility data sourced from public OSM — already public
- ✅ Triage bot outputs are ephemeral — no conversation storage

Healthcare access inequity is a real problem. This tool is designed to support **evidence-based public health planning**, not surveillance or data collection.

---

## 🚢 Deployment

| Platform | URL | Purpose |
|---|---|---|
| Streamlit Community Cloud | [Add link after deploy] | Interactive dashboard |
| Hugging Face Spaces | [Add link after deploy] | Triage bot demo |

---

## 📹 Demo

> [Add demo video link here after recording]

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/`
4. Open a pull request

---

## 📄 License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

*Built as a portfolio project demonstrating geospatial analysis, optimization, and responsible AI for public health.*
