# 📈 Tech Skill Forecaster: Hacker News Edition

An end-to-end data pipeline and machine learning project that extracts, analyzes, and forecasts tech hiring trends using the **Hacker News "Who is Hiring"** monthly ecosystem.

**Track the demand for programming languages, frameworks, and specialized roles in real-time. Predict future tech skill market share using exponential smoothing forecasts.**

---

## 🎯 Quick Links

- **[Architecture Overview](#-architecture)** — 4-phase ETL pipeline
- **[Getting Started](#-getting-started)** — Installation & quickstart
- **[Dashboard](#-interactive-dashboard)** — Streamlit visualization
- **[Insights](#-key-insights)** — Current trends (March 2026)

---

## 🚀 Project Overview

This autonomous system processes monthly job postings from Hacker News "Who is Hiring" threads to track and forecast the demand for tech skills. The architecture combines:

- **NLP & Skill Recognition** using ESCO taxonomy engine
- **Multi-stage data normalization** with skill blacklist filtering
- **Role classification** (Frontend/Backend/DevOps/ML/Data Engineer, etc.)
- **Market share analysis** across skill categories
- **Time-series forecasting** using exponential smoothing
- **Interactive dashboard** for exploration and insights

- **Current Coverage**: 200+ job postings | Starting January 2026
- **Skills Tracked**: 30+ programming languages, frameworks, and tools
- **Roles Identified**: 3+ specialized role categories

---

## 📐 Architecture

### 4-Phase ETL Pipeline

```
Phase 1: EXTRACTION        Phase 2: TRANSFORMATION       Phase 3: PREDICTION         Phase 4: VISUALIZATION
         ↓                         ↓                              ↓                              ↓
  Raw Hiring Posts    →   NLP + Skill Mining    →   Trend Aggregation    →    Interactive Dashboard
  (JSON)                  (ESCO Taxonomy)           (Growth Momentum)          (Streamlit App)
                          ↓
                    Normalization & Roles
                    (Classifier)
                          ↓
                    Processed CSV
                    (ML-Ready)
```

### Phase Details

#### **Phase 1: Extraction** (`src/extraction/extract.py`)

- Scrapes Hacker News "Who is Hiring" monthly threads
- Saves raw job postings as JSON
- Storage: `data/raw/year=YYYY/month=MM/`

#### **Phase 2: Transformation** (`src/transformation/`)

- **`esco_engine.py`**: ESCO (European Skills/Competencies/Occupations) taxonomy engine
  - Matches job text against 1000+ digital skill definitions
  - Prevents false positives (e.g., "Java" not in "JavaScript")
  - Single-pass extraction with word-boundary protection
- **`transform.py`**: Main NLP pipeline
  - BeautifulSoup for HTML parsing
  - spaCy for NER (Named Entity Recognition)
  - Extracts skills from raw posting text
  - Output: `data/processed/year=YYYY/month=MM/NLP_extracted.csv`

- **`normalizer.py`**: Cleaning & role classification
  - 200+ skill normalization rules
  - Blacklist filtering (roles, companies, generic terms)
  - Role classification heuristics (Frontend/Backend/Full Stack/DevOps/ML/Data)
  - Quality reporting (data coverage %)
  - Output: `data/processed/year=YYYY/month=MM/ml_ready.csv` + `dq_report.json`

#### **Phase 3: Prediction** (`src/prediction/`)

- **`aggregator.py`**: Trend aggregation engine
  - Calculates global market share per skill per month
  - Computes role-specific trends (skills needed by role)
  - Month-over-month growth momentum
  - Detects "breakout tech" (top gainers)
  - Output: `data/analysis/latest_trends.csv`, `growth_momentum.csv`

- **`forecaster.py`**: Time-series forecasting
  - Exponential smoothing (Holt-Winters method)
  - Generates 1-month ahead predictions
  - Classifies trends (Increasing/Decreasing/Stable)
  - Output: `data/analysis/forecasts.csv`

#### **Phase 4: Dashboard** (`app/dashboard.py`)

- **Interactive Streamlit app** with:
  - Real-time skill popularity metrics
  - Future skill forecasts (with growth predictions)
  - Global trend lines (historical market share)
  - Month-over-month momentum charts
  - Customizable skill filters & period selection

---

## 📂 Project Structure

```
tech-skill-forecaster/
├── app/
│   └── dashboard.py                          # Streamlit interactive dashboard
├── data/
│   ├── raw/year=YYYY/month=MM/
│   │   └── *.json                           # Raw job postings (immutable)
│   ├── processed/year=YYYY/month=MM/
│   │   ├── NLP_extracted.csv                # Extracted skills (before normalization)
│   │   ├── ml_ready.csv                     # Normalized, role-classified
│   │   └── dq_report.json                   # Data quality metrics
│   ├── external/esco/
│   │   ├── skills_en.csv                    # ESCO skill taxonomy
│   │   └── DigitalSkill_en.csv              # ESCO digital skills filter
│   └── analysis/
│       ├── latest_trends.csv                # Current month skill market share
│       ├── growth_momentum.csv              # Month-over-month growth %
│       ├── forecasts.csv                    # Next-month predictions
│       └── snapshot_YYYY-MM-DD/             # Timestamped snapshots
│           ├── aggregated_trends.csv
│           └── role_trends.csv
├── src/
│   ├── extraction/
│   │   └── extract.py                       # Phase 1: Scraping & collection
│   ├── transformation/
│   │   ├── transform.py                     # Phase 2a: NLP extraction
│   │   ├── esco_engine.py                   # ESCO taxonomy matching
│   │   └── normalizer.py                    # Phase 2b: Cleaning & role classification
│   └── prediction/
│       ├── aggregator.py                    # Phase 3a: Trend aggregation
│       └── forecaster.py                    # Phase 3b: Time-series forecasting
├── notebooks/
│   ├── 01_analysis.ipynb                    # Exploratory analysis
│   └── 01_trends.ipynb                      # Trend visualization demo
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 🛠 Getting Started

### Prerequisites

- Python 3.8+
- Conda or pip (see `requirements.txt`)

### Installation

```bash
# Clone the repository
cd tech-skill-forecaster

# Create a virtual environment (recommended)
micromamba create -n dsml python=3.11
micromamba activate dsml

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (for NER)
python -m spacy download en_core_web_sm
```

### Dependencies

| Category             | Packages                            |
| -------------------- | ----------------------------------- |
| **Data Collection**  | `requests`, `pathlib`               |
| **Data Processing**  | `pandas`, `beautifulsoup4`, `spacy` |
| **ML & Forecasting** | `scikit-learn`, `numpy`             |
| **Visualization**    | `streamlit`, `plotly`               |

### Running the Pipeline

Each phase can be run independently:

```bash
# Phase 1: Extract hiring data from Hacker News (run monthly)
python src/extraction/extract.py

# Phase 2: Transform and Normalizer raw text into normalized skills
python src/transformation/transform.py
python src/transformation/normalizer.py

# Phase 3: Aggregate trends and generate forecasts
python src/prediction/aggregator.py
python src/prediction/forecaster.py

# Phase 4: Launch the interactive dashboard
streamlit run app/dashboard.py
```

---

## 📊 Interactive Dashboard

The Streamlit app provides real-time visualization of:

### **Metric Cards**

- 🏆 **Top Skill** — Most mentioned tech in current month
- 🔥 **Top Grower** — Fastest-growing skill (month-over-month)
- 📈 **Skills Tracked** — Total unique skills in current dataset

### **Future Forecast Section**

- 📡 Predicted market share for next month
- 🎯 Highest expected gainers
- ↗️ Trend classification (Increasing/Decreasing/Stable)
- 🔄 Regenerate forecasts on demand

### **Global Trends Chart**

- 📉 Market share evolution over time
- 🎨 Multi-skill comparison
- 🔍 Interactive period filtering

### **Growth Momentum Chart**

- 📊 Month-over-month growth rates
- 📆 Faceted by period
- 🔢 Optional value labels
- ⏭️ Hide zero-growth periods

---

## 🔍 Key Insights (Current Data)

Based on the February 2026 data snapshot:

### **Top 10 Skills by Market Share**

| Rank | Skill            | Market Share | Trend |
| ---- | ---------------- | ------------ | ----- |
| 1    | AI               | 15.65%       | ↑     |
| 2    | TypeScript       | 7.67%        | ↓     |
| 3    | Python           | 7.67%        | ↓     |
| 4    | JavaScript       | 8.14%        | ↓     |
| 5    | React            | 6.73%        | ↑     |
| 6    | AWS              | 5.16%        | ↓     |
| 7    | PostgreSQL       | 3.44%        | ↓     |
| 8    | Machine Learning | 3.60%        | ↑     |
| 9    | Go               | 3.13%        | ↑     |
| 10   | Rust             | 2.97%        | ↓     |

### **Breakout Tech (March 2026 Forecast)**

- **LLM** — +0.48% (highest momentum)
- **Node.js** — +0.48%
- **Kubernetes** — +0.39%
- **React** — +0.55% (among frameworks)

### **Role Distribution**

- Backend Engineers — 32%
- Full Stack Engineers — 28%
- Frontend Engineers — 15%
- DevOps Engineers — 8%
- ML/AI Engineers — 7%
- Other specialized roles — 10%

### **Skill Categories**

- **Languages** — Python, JavaScript, TypeScript, Go, Rust
- **Databases** — PostgreSQL, MongoDB, Redis, Snowflake
- **Cloud** — AWS, GCP, Azure
- **DevOps** — Docker, Kubernetes, Terraform
- **ML/AI** — PyTorch, TensorFlow, LLM, Transformers

---

## 🔄 Data Flow

### Example: Processing January 2026 Data

1. **Raw Extract** → `data/raw/year=2026/month=01/2026-01_raw_hiring.json`
   - ~Job postings from HN "Who is Hiring?" thread

2. **NLP Extraction** → `data/processed/year=2026/month=01/NLP_extracted.csv`
   - Extracts 5-15 skills per posting using ESCO taxonomy
   - Quality: ~78% coverage (posts with ≥1 skill)

3. **Normalization** → `data/processed/year=2026/month=01/ml_ready.csv`
   - Standardizes skill names (e.g., "js" → "JavaScript")
   - Filters 200+ blacklisted terms (CEO, Agile, Salesforce, etc.)
   - Classifies roles (Frontend/Backend/Full Stack/etc.)
   - Output: 2,340 clean rows with skills + roles

4. **Aggregation** → `data/analysis/latest_trends.csv`
   - Calculates market share: (skill_count / total_posts) × 100
   - Computes role-specific trends
   - Records monthly snapshot

5. **Forecasting** → `data/analysis/forecasts.csv`
   - Fits exponential smoothing to 2-month history
   - Generates next-month predictions
   - Labels trends: Increasing (+), Decreasing (-), Stable (=)

6. **Visualization** → Streamlit Dashboard
   - User selects periods/skills for custom analysis
   - Interactive charts with Plotly

---

## 🛡️ Key Features

### **Smart Skill Extraction**

- **ESCO Taxonomy Engine** — 1,000+ digital skill definitions
- **Word-Boundary Protection** — "Java" ≠ "JavaScript"
- **Synonym Mapping** — "PyTorch" and "pytorch" → standardized

### **Comprehensive Normalization**

- **200+ Blacklist Terms** — Filters out noise (roles, companies, generic terms)
- **9 Skill Categories** — Languages, Web, Frontend, Backend, Cloud, Databases, DevOps, ML/AI
- **12 Role Classifications** — Frontend/Backend/Full Stack/DevOps/Cloud/Data/ML/QA/Security/etc.

### **Data Quality**

- **Coverage Reports** — % of posts with ≥1 valid skill extracted
- **Skill Filtering** — Only skills appearing 2+ times or in mapping
- **Role Heuristics** — Context-aware detection (text + skills)

### **Forecasting**

- **Exponential Smoothing** — Holt-Winters model for trend continuation
- **1-Month Lookahead** — Predictions for next month
- **Trend Classification** — Automatic labeling (Increasing/Decreasing/Stable)

### **Interactive Dashboard**

- **Streamlit App** — Web-based, zero-deployment
- **Real-time Reload** — Click "Regenerate" to update forecasts instantly
- **Custom Filtering** — Select periods, skills, trend types
- **Plotly Charts** — Interactive hover, export as PNG

---

## 📝 Configuration & Tuning

### **Skill Filtering** (`src/prediction/aggregator.py`)

```python
config = AggregatorConfig(
    min_skill_posts=5,        # Require ≥5 mentions to include skill
    top_breakout_skills=3,    # Report top 3 breakout skills
    chunk_size=50000          # Memory-efficient file processing
)
```

### **Skill Normalization** (`src/transformation/normalizer.py`)

Add custom skill mappings:

```python
skill_mapping = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "custom-tool": "Custom Tool",
}
normalizer.norm_map.update(skill_mapping)
```

---

## 📚 Technologies Used

| Layer               | Technology                  | Purpose                            |
| ------------------- | --------------------------- | ---------------------------------- |
| **Data Collection** | Python, Requests            | Web scraping                       |
| **NLP**             | spaCy, NLTK, BeautifulSoup4 | Text processing, entity extraction |
| **Taxonomy**        | Aho-Corasick (ahocorasick)  | Efficient keyword matching         |
| **Data Processing** | Pandas                      | ETL & aggregation                  |
| **ML/Forecasting**  | scikit-learn, NumPy         | Statistical modeling               |
| **Visualization**   | Streamlit, Plotly           | Interactive dashboard              |
| **Infrastructure**  | Python, Pathlib, Logging    | Core utilities                     |

---

## 🚦 Project Status

- ✅ Phase 1: Data extraction (Jan-Feb 2026)
- ✅ Phase 2: NLP + normalization (working, 78% coverage)
- ✅ Phase 3: Trend aggregation & forecasting (production)
- ✅ Phase 4: Dashboard (live, real-time)
- 🔄 **Next**: Multivariate forecasting, confidence intervals
- 📍 **Planned**: Skill dependency graph, role-based recommendations

---

## 📖 Usage Examples

### **1. View Latest Trends**

```python
import pandas as pd

trends = pd.read_csv("data/analysis/latest_trends.csv")
print(trends.nlargest(10, "market_share")[["skills", "market_share"]])
```

### **2. Check Skill Momentum**

```python
momentum = pd.read_csv("data/analysis/growth_momentum.csv")
print(momentum.iloc[-1].nlargest(5))  # Top 5 gainers last month
```

### **3. Analyze Forecast**

```python
forecast = pd.read_csv("data/analysis/forecasts.csv")
gainers = forecast[forecast["trend"] == "Increasing"].sort_values("delta_share", ascending=False)
print(gainers[["skill", "current_share", "predicted_share", "delta_share"]])
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Multi-variate forecasting (ARIMA, Prophet)
- [ ] Confidence intervals for predictions
- [ ] Skill dependency/correlation analysis
- [ ] Role-based skill recommendations
- [ ] Historical data archive (pre-2026)
- [ ] API endpoint for programmatic access
- [ ] Production deployment (Docker, cloud)

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute.

---

## 📞 Contact & Support

For questions or issues: xxbayu9@gmail.com

- Review the [notebooks](notebooks/) for exploratory analysis examples
- Check [data quality reports](data/processed/*/dq_report.json) for coverage metrics
- Inspect [snapshot analysis](data/analysis/snapshot_*/) for monthly insights

---

## 🙏 Acknowledgments

- **ESCO Taxonomy** — European digital skills taxonomy for skill matching
- **Hacker News** — Monthly "Who is Hiring?" community posts
- **Streamlit** — Interactive dashboard framework
- **scikit-learn** — Forecasting and statistical modeling

---

**Last Updated**: March 1, 2026 | **Data Snapshot**: February 2026
