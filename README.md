# 📈 Tech Skill Forecaster: Hacker News Edition

An autonomous data pipeline and machine learning project designed to extract, analyze, and forecast labor market trends using the **Hacker News "Who is Hiring"** ecosystem.

## 🚀 Project Overview

This system tracks the demand for programming languages, frameworks, and specialized roles (like AI/ML Engineers) by processing monthly job postings. It utilizes a multi-phase ETL (Extract, Transform, Load) architecture to turn raw text into predictive insights.

## 📂 Project Structure

```text
tech-skill-forecaster/
├── data/
│   ├── raw/                # Immutable JSON source data
│   └── processed/          # Cleaned datasets for ML (Phase 2)
├── src/
│   ├── extraction/         # Phase 1: Autonomous API Discovery & Fetching
│   ├── transformation/      # Phase 2: NLP & Skill Extraction logic
│   └── prediction/         # Phase 3: Time-series forecasting models
├── app/                    # Phase 4: Streamlit Dashboard
├── requirements.txt        # Project dependencies
└── README.md
```
