
import json
import argparse
from pretraitement_nlp.pipeline import run_full_pipeline

def main():
    print('Starting ...')
    parser = argparse.ArgumentParser(description='Run the NLP pipeline using a config file.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    input_path = config.get('input_path')
    output_dir = config.get('output_dir', 'results')
    launch_dashboard = config.get('launch_dashboard', False)

    run_full_pipeline(input_path, output_dir, launch_dashboard)

if __name__ == '__main__':
    main()
