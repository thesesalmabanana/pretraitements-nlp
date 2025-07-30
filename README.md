
# NLP Topic Modeling Preprocessing Pipeline

This repository contains a robust and modular pipeline for **text preprocessing**, **topic modeling** with BERTopic, and **interactive result exploration** via a Streamlit dashboard.  
It is designed for high-throughput, reproducible experiments on raw text data (e.g., surveys, user feedback, or transcripts).

---

## 🧰 Features

- 📄 Preprocessing of raw textual data (cleaning, normalization, lemmatization, etc.)
- 🔎 Topic modeling using BERTopic (with KMeans-based variants)
- 📁 Fully automated file structure with time-stamped outputs
- 📊 Interactive dashboard for in-depth topic exploration
- 🔧 Configurable pipeline via `config.json`
- ⚙️ Easily extendable and modular codebase
- ✅ MIT License — open to academic and commercial reuse

---

## 🚀 Usage

### 1. Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure your pipeline

```json
{
  "input_path": "input/my_data.csv",
  "output_dir": "results",
  "launch_dashboard": true
}
```

- `input_path`: path to your CSV file (must contain a `comment` column)
- `output_dir`: root folder where outputs will be saved
- `launch_dashboard`: if `true`, opens the Streamlit dashboard in your browser at the end of processing

---

### 3. Run the pipeline

```bash
python cli.py
```

This will:

- Clean your data
- Run BERTopic with multiple values of `k`
- Save results and visualizations
- Optionally launch the dashboard

---

## 📊 Dashboard Overview

If `launch_dashboard = true`, the pipeline ends by launching a **Streamlit dashboard** that provides:

### Sidebar

- Selection of results folder
- Dynamic dropdown to select number of topics (`k`)
- Stop button to exit cleanly

### Main Interface

- 📌 Pipeline summary: number of documents, topics, unassigned documents
- 🔍 Search: filter documents by keyword
- 🧠 Topic explorer: view documents assigned to a given topic
- 📈 Topic summary table
- 🌐 Interactive visualizations:
  - `topics_visualization.html` (global clustering)
  - `IDM_kX.html` (specific to selected `k`)

---

## 📁 Output Structure

After each run, a new subfolder is created under `results/YYYY-MM-DD/HH-MM-SS/` containing:

```
results/
└── 2025-07-30/
    └── 15-42-10/
        ├── cleaned/
        │   └── cleaned_data.csv
        ├── topics/
        │   ├── topics.csv
        │   ├── summary_topics.csv
        │   └── topics_visualization.html
        ├── analytics/
        │   └── k10/
        │       ├── summary_topics.csv
        │       └── IDM_k10.html
        └── config_used.json
```

---

## 🧪 Requirements

- Python ≥ 3.8
- `bertopic`, `sentence-transformers`, `streamlit`, `scikit-learn`, `pandas`, `umap-learn`, `matplotlib`

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and redistribute it for academic and commercial purposes. See [`LICENSE`](LICENSE) for details.

---

## 🤝 Contributions

Contributions are welcome. Please submit pull requests with detailed commit messages and test your changes locally.
