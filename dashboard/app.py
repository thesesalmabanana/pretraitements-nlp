import streamlit as st
import pandas as pd
import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="NLP Topic Explorer", layout="wide")

st.title("NLP Topic Modeling Dashboard")

def find_latest_results(base_path="results"):
    if not os.path.exists(base_path):
        return ""
    dates = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if not dates:
        return ""
    latest_date = dates[-1]
    date_path = os.path.join(base_path, latest_date)
    times = sorted([t for t in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, t))])
    if not times:
        return ""
    latest_time = times[-1]
    return os.path.join(date_path, latest_time)

# Auto-load latest results
default_results_path = find_latest_results()

# Sidebar config
st.sidebar.header("Configuration")
results_path = st.sidebar.text_input("Results folder path", default_results_path)

# Stop button
if st.sidebar.button("❌ Stop and Exit"):
    st.warning("Stopping the Streamlit server...")
    time.sleep(1)
    os._exit(0)

# Check if path exists
if not os.path.exists(results_path):
    st.error(f"No results found at: {results_path}")
    st.stop()

# Display analytics/kXX selection dynamically
analytics_path = os.path.join(results_path, "analytics")
available_k = sorted([d for d in os.listdir(analytics_path) if d.startswith("k")]) if os.path.exists(analytics_path) else []
selected_k = st.sidebar.selectbox("Select number of topics (k)", available_k) if available_k else None

# Load topics.csv
topics_csv = os.path.join(results_path, "topics", "topics.csv")
df_topics = pd.read_csv(topics_csv) if os.path.exists(topics_csv) else None

# Load summary_topics.csv for selected k
summary_csv = os.path.join(results_path, "analytics", selected_k, "summary_topics.csv") if selected_k else None
df_summary = pd.read_csv(summary_csv) if summary_csv and os.path.exists(summary_csv) else None

# HEADER SUMMARY
st.subheader("Pipeline Summary")
if df_topics is not None:
    total_docs = len(df_topics)
    num_topics = df_topics["topic"].nunique()
    topic_counts = df_topics["topic"].value_counts()
    avg_size = topic_counts.mean()
    unassigned = (df_topics["topic"] == -1).sum()

    st.markdown(
        f"- **Total documents**: {total_docs}\n"
        f"- **Topics discovered**: {num_topics}\n"
        f"- **Avg. topic size**: {avg_size:.2f}\n"
        f"- **Unassigned documents**: {unassigned}"
    )

# SELECT TOPIC TO EXPLORE
if df_topics is not None:
    st.subheader("Explore documents by topic")
    topic_ids = sorted(df_topics["topic"].dropna().unique())
    selected_topic = st.selectbox("Select a topic ID", topic_ids)
    st.write(df_topics[df_topics["topic"] == selected_topic][["comment"]].head(20))

# SEARCH DOCUMENTS
if df_topics is not None:
    st.subheader("Search for a keyword")
    query = st.text_input("Enter keyword")
    if query:
        mask = df_topics["comment"].str.contains(query, case=False, na=False)
        st.write(f"{mask.sum()} results found")
        st.dataframe(df_topics[mask][["comment", "topic"]].head(20))

# DISPLAY TOPIC SUMMARY
if df_summary is not None:
    st.subheader(f"Topic Summary ({selected_k})")
    st.dataframe(df_summary)

# DISPLAY GLOBAL BERTopic VISUALIZATION
html_global_path = os.path.join(results_path, "topics", "topics_visualization.html")
if os.path.exists(html_global_path):
    st.subheader("Global BERTopic Visualization (all topics)")
    with open(html_global_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=660, scrolling=False)

# DISPLAY DYNAMIC VISUALIZATION FOR SELECTED k
if selected_k:
    html_dynamic_path = os.path.join(results_path, "analytics", selected_k, f"IDM_{selected_k}.html")
    if os.path.exists(html_dynamic_path):
        st.subheader(f"BERTopic Visualization for {selected_k}")
        with open(html_dynamic_path, "r", encoding="utf-8") as f:
            html_dynamic_content = f.read()
        st.components.v1.html(html_dynamic_content, height=660, scrolling=False)
    else:
        st.info(f"No visualization HTML found for {selected_k}")
