
import os
import datetime
import json
import subprocess
from pretraitement_nlp.cleaning import run_cleaning_pipeline
from pretraitement_nlp.topic_modeling import run_topic_modeling_pipeline
from pretraitement_nlp.analytics import run_analytics_pipeline

def make_output_dir(base_dir='results') -> str:
    now = datetime.datetime.now()
    date_folder = os.path.join(base_dir, now.strftime('%Y-%m-%d'))
    os.makedirs(date_folder, exist_ok=True)
    time_folder = os.path.join(date_folder, now.strftime('%H-%M-%S'))
    os.makedirs(time_folder, exist_ok=True)
    return time_folder

def run_full_pipeline(input_path: str, base_output: str = 'results', launch_dashboard: bool = False):
    print('Starting NLP preprocessing pipeline...')
    output_dir = make_output_dir(base_output)
    cleaned_path = os.path.join(output_dir, 'cleaned.csv')

    print('Step 1: Cleaning raw input data')
    run_cleaning_pipeline(input_path, cleaned_path)
    print(f'Cleaned data saved to: {cleaned_path}')

    print('Step 2: Running topic modeling')
    topic_dir = os.path.join(output_dir, 'topics')
    run_topic_modeling_pipeline(cleaned_path, topic_dir)
    print(f'Topic modeling results saved to: {topic_dir}')

    print('Step 3: Running topic analytics for multiple k')
    analytics_dir = os.path.join(output_dir, 'analytics')
    run_analytics_pipeline(cleaned_path, analytics_dir)
    print(f'Analytics results saved to: {analytics_dir}')

    print('Pipeline execution completed.')

    if launch_dashboard:
        print('Launching Streamlit dashboard...')
        subprocess.run(['streamlit', 'run', 'dashboard/app.py', '--', '--results', output_dir])
