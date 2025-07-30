import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def run_topic_modeling_pipeline(input_path: str, output_dir: str):
    # Load pre-cleaned data
    df = pd.read_csv(input_path)
    texts = df['cleaned'].dropna().tolist()

    # Create sentence embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Create and train BERTopic model with UMAP disabled (avoids eigh errors)
    topic_model = BERTopic(language="english", umap_model=None, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Save topic assignments
    df['topic'] = topics
    df.to_csv(os.path.join(output_dir, "topics.csv"), index=False)

    # Save topic summary
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, "summary_topics.csv"), index=False)

    # Try to generate BERTopic interactive HTML visualization
    if len(texts) >= 3:
        try:
            # Some BERTopic versions use c_tf_idf for visualization by default
            # If it's a sparse matrix, convert it to dense
            if hasattr(topic_model.c_tf_idf_, 'toarray'):
                dense_embeddings = topic_model.c_tf_idf_.toarray()
            else:
                dense_embeddings = topic_model.c_tf_idf_

            fig = topic_model.visualize_topics(custom_embeddings=dense_embeddings)
            fig.write_html(os.path.join(output_dir, "topics_visualization.html"))

        except Exception as e:
            print(f"[WARNING] Could not generate interactive topic visualization: {e}")
    else:
        print("[INFO] Not enough documents to generate visualization (minimum = 3)")
