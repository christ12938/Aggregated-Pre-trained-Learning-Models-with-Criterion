#!/bin/bash

python3 pickle_json_converter.py -i data/scidocs_vocab.pkl -o data/scidocs_vocab.json -m vocab
python3 pickle_json_converter.py -i data/scidocs_doc.pkl -o data/scidocs_doc.json -m doc

python3 pickle_json_converter.py -i data/amazon_vocab.pkl -o data/amazon_vocab.json -m vocab
python3 pickle_json_converter.py -i data/amazon_doc.pkl -o data/amazon_doc.json -m doc

python3 pickle_json_converter.py -i data/french_news_vocab.pkl -o data/french_news_vocab.json -m vocab
python3 pickle_json_converter.py -i data/french_news_doc.pkl -o data/french_news_doc.json -m doc

python3 pickle_json_converter.py -i data/merged_vocab.pkl -o data/merged_vocab.json -m vocab
python3 pickle_json_converter.py -i data/merged_doc.pkl -o data/merged_doc.json -m doc

