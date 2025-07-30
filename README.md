# Pretraitement NLP 🧠

This project provides a modular and extensible **NLP preprocessing pipeline** for the cleaning, topic modeling, and analysis of Reddit-style comment datasets.

The pipeline is structured around **best software engineering practices**, including modularity, testability, and configuration-driven execution.

---

## 📌 Features

- **Cleaning pipeline**: text normalization, lemmatization (spaCy), stopwords filtering (custom + NLTK)
- **Topic modeling**: using BERTopic + Sentence Transformers
- **Topic analytics**: evaluation across multiple topic numbers via KMeans
- **Modular architecture**: every processing step is decoupled in its own module
- **CLI interface**: run entire pipeline via a config file
- **Automatic results folder**: timestamped output directory
- **GitHub-ready**: includes `.gitignore`, `requirements.txt`, `README`, `config.json.exemple`

---

## 🧭 Project Structure

```
pretraitement_nlp/
├── cleaning.py               # Data cleaning module
├── topic_modeling.py         # BERTopic modeling module
├── analytics.py              # Topic analysis via KMeans
├── pipeline.py               # Pipeline orchestration
├── __init__.py               # (empty or optional)
cli.py                        # CLI script entrypoint
config.json.exemple           # Sample configuration
inputs/                       # Input CSVs (user-provided)
results/                      # Output folder (auto-organized by date)
requirements.txt              # Python dependencies
README.md                     # This documentation
.gitignore                    # Git exclusions
```

---

## ⚙️ Configuration

The pipeline uses a single `config.json` file for all input/output paths.

Create your own `config.json` by copying the template:

```bash
cp config.json.exemple config.json
```

Then edit it:

```json
{
    "input_path": "inputs/comments.csv",
    "output_dir": "results"
}
```

---

## 🚀 Usage

1. Install dependencies (in a virtualenv recommended):

```bash
pip install -r requirements.txt
```

2. Place your raw comment data (CSV with a column `comment`) in the `inputs/` folder.

3. Run the pipeline:

```bash
python cli.py
```

By default, this uses `config.json` at the project root. You can specify another file:

```bash
python cli.py --config path/to/config.json
```

---

## 🧼 Cleaning Logic

- Lowercasing
- Removal of punctuation/special characters
- Lemmatization using `spaCy` (`en_core_web_sm`)
- Stopwords removed using:
  - NLTK default English stopwords
  - Custom set (e.g., "https", ".com", etc.)

---

## 🧠 Topic Modeling

Uses [BERTopic](https://maartengr.github.io/BERTopic/) with `SentenceTransformer("all-MiniLM-L6-v2")`.

- Fit topic model using contextual embeddings
- Save HTML visualization of topics
- Export per-document topic assignments

---

## 📊 Topic Analytics

Evaluate different values of `k` for topic clustering (e.g., 8–15):

- Applies `KMeans(n_clusters=k)`
- Fits `BERTopic` with custom clustering
- Generates `IDM_k{k}.html` and CSV summary for each `k`

---

## 🧪 Future Extensions

- Add unit tests for each module
- Deploy as a Python package (`setup.py` or `pyproject.toml`)
- Add language support / multilingual embeddings
- Create a Streamlit dashboard to explore results

---

## 👤 Authors

- **Original scripts**: Provided by project owner
- **Refactoring and modularization**: GPT-4 (OpenAI) under PhD instruction context

---

## 📁 License

This project is released under a license of your choice (e.g., MIT, Apache-2.0).

