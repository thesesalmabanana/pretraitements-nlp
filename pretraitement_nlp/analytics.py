import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_text_data(path: str, column: str = 'cleaned') -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[[column]].dropna()

def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True)

def test_bertopic_with_kmeans(df, embeddings, k_values, output_dir):
    texts = df.iloc[:, 0].tolist()
    for k in k_values:
        if k >= len(texts):
            print(f"Skipping k={k}: k >= number of documents ({len(texts)})")
            continue

        print(f"Testing KMeans with k = {k}")
        kmeans = KMeans(n_clusters=k, random_state=42)

        # Désactivation explicite de UMAP
        topic_model = BERTopic(
            language='english',
            calculate_probabilities=False,
            verbose=False,
            umap_model=None,
            hdbscan_model=kmeans
        )

        topics, _ = topic_model.fit_transform(texts, embeddings)

        out_dir = os.path.join(output_dir, f'k{k}')
        os.makedirs(out_dir, exist_ok=True)

        # Sauvegarde des fichiers
        topic_model.visualize_topics().write_html(os.path.join(out_dir, f'IDM_k{k}.html'))
        topic_model.get_topic_info().to_csv(os.path.join(out_dir, 'summary_topics.csv'), index=False)

def run_analytics_pipeline(input_path: str, output_dir: str, k_values=[8, 9, 10, 11, 12, 13, 14, 15]):
    df = load_text_data(input_path)
    embeddings = generate_embeddings(df.iloc[:, 0].tolist())
    test_bertopic_with_kmeans(df, embeddings, k_values, output_dir)
