# Morocco Healthcare Optimization

[![CI](https://github.com/marouane-ouaomar/morocco-healthcare-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/marouane-ouaomar/morocco-healthcare-optimization/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

> A geospatial healthcare access analysis and optimization platform for Morocco.
> Built for data scientists, GIS analysts, and public health professionals.

---

## What This Project Does

HealthAccess Morocco maps **where healthcare facilities exist**, measures **how accessible they are** to the population, identifies **underserved regions**, and simulates the impact of **adding new facilities or mobile health units**.

| Layer | What it answers |
|---|---|
| **Facility Map** | Where are hospitals, clinics, pharmacies, and doctors in Morocco? |
| **Access Metrics** | What % of the population lives within 5 / 10 / 20 km of a facility? |
| **Optimization** | Where should new facilities be placed to maximize coverage? |
| **Triage Bot** | Can a safe, rule-based AI help patients know when to seek emergency care? |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/marouane-ouaomar/morocco-healthcare-optimization.git
cd morocco-healthcare-optimization

# 2. Set up Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fetch facility data for Morocco (optional — processed data is included)
python scripts/fetch_osm_facilities.py
python -c "from src.data_prep import run_pipeline; run_pipeline()"

# 5. Run the Streamlit dashboard
streamlit run webapp/streamlit_app.py
```

---

## Project Structure

```
morocco-healthcare-optimization/
├── src/
│   ├── data_prep.py          # Facility & population data cleaning
│   ├── access_metrics.py     # KD-tree nearest-neighbor, coverage ratios
│   ├── kmeans_placement.py   # Weighted KMeans optimization
│   ├── scenario_simulator.py # Scenario runner & output
│   ├── facility_locator.py   # Nearest-facility lookup (triage bot)
│   ├── spatial_utils.py      # Morocco boundary & spatial integrity
│   └── triage_engine.py      # Safe rule-based triage bot
├── scripts/
│   └── fetch_osm_facilities.py  # Overpass API → GeoJSON
├── data/
│   ├── raw/                  # Raw OSM fetch (gitignored, regenerated)
│   ├── processed/            # Clean GeoJSON (committed for cloud deploy)
│   └── boundaries/           # Morocco admin boundary
├── tests/                    # pytest test suite
├── docs/
│   └── SAFETY.md             # Triage bot safety documentation
├── webapp/
│   ├── streamlit_app.py      # Main dashboard (4 tabs)
│   └── pages/
│       └── triage_bot.py     # Triage assistant (multi-page app)
├── packages.txt              # System deps for Streamlit Community Cloud (GDAL)
├── .streamlit/config.toml    # Theme & server config
├── .github/workflows/ci.yml  # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## Dashboard Tabs

The Streamlit app is organized into four dedicated tabs:

| Tab | Contents |
|---|---|
| **Dashboard** | Interactive facility map with population density heatmap and symbol-matched legend |
| **Accessibility Metrics** | Coverage KPIs, regional pop/facility ratios, facility type breakdown |
| **Scenario Simulation** | Intervention controls, before/after metrics, proposed site map |
| **Data Export** | Download metrics CSV, scenario JSON, filtered facilities |

---

## Data Sources

| Dataset | Source | License |
|---|---|---|
| Healthcare facilities | [OpenStreetMap](https://www.openstreetmap.org/) via [Overpass API](https://overpass-api.de/) | ODbL |
| Population grid | [WorldPop](https://www.worldpop.org/) (Morocco 100m grid) | CC BY 4.0 |
| Administrative boundaries | [GADM](https://gadm.org/) / OpenStreetMap | See source |

> Raw population rasters (TIFF) are not committed. A synthetic population grid is included for demos.

### Current Dataset (OSM sync)

| Facility type | Count |
|---|---|
| Pharmacy | 6,924 |
| Doctor | 524 |
| Hospital | 493 |
| Clinic | 257 |
| **Total** | **8,198** |

---

## Running the Full Pipeline

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

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Set the **main file path** to `webapp/streamlit_app.py`.
4. Ensure `packages.txt` and `requirements.txt` are at the repository root (they are).
5. Commit `data/processed/facilities.geojson` and `data/processed/popgrid.geojson` — these are required at runtime and are **not** gitignored.
6. (Optional) Add secrets in the Streamlit Cloud dashboard:
   - `ANTHROPIC_API_KEY` — enables LLM-powered triage responses
   - `OSRM_SERVER_URL` — enables road-network distance calculations

The app runs fully without any API keys. The triage bot falls back to local rule-based responses.

### Local secrets

Copy `.env.example` to `.env` for local development. Never commit `.env`.

---

## Triage Bot

The triage assistant is a **research demo only**.

- Detects emergencies: chest pain, breathing difficulty, stroke signs, severe bleeding
- Outputs only structured JSON advice (no diagnosis)
- Always recommends professional consultation
- Local fallback mode — no API key required for demos
- Not a medical device
- Not a substitute for clinical care

See [`docs/SAFETY.md`](docs/SAFETY.md) for the full safety specification.

Access via the sidebar page navigation: **Triage Bot**.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage targets:

- Data cleaning functions (synthetic fixtures)
- Access metric calculations (known-answer tests)
- Scenario simulator (synthetic population grid)
- Triage emergency detection (100-case synthetic set)

---

## Ethics & Data Privacy

This project handles **no personal data**. Specifically:

- Only aggregated population grids (no individual records)
- No names, addresses, or identifiers linked to individuals
- No sensitive medical data stored or transmitted
- Facility data sourced from public OSM — already public
- Triage bot outputs are ephemeral — no conversation storage

Healthcare access inequity is a real problem. This tool is designed to support **evidence-based public health planning**, not surveillance or data collection.

---

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/`
4. Open a pull request

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

*Built as a portfolio project demonstrating geospatial analysis, optimization, and responsible AI for public health.*
