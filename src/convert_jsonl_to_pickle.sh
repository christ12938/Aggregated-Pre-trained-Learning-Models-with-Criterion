#!/bin/bash

python3 pickle_json_converter.py -i data/scidocs_scores.jsonl -o data/scidocs_scores.pkl -m jsonl

python3 pickle_json_converter.py -i data/amazon_scores.jsonl -o data/amazon_scores.pkl -m jsonl

python3 pickle_json_converter.py -i data/french_news_scores.jsonl -o data/french_news_scores.pkl -m jsonl

python3 pickle_json_converter.py -i data/merged_scores.jsonl -o data/merged_scores.pkl -m jsonl
