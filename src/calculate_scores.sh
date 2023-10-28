#!/bin/bash

time ./calculate_scores -v data/scidocs_vocab.json -d data/scidocs_doc.json -o data/scidocs_scores.jsonl -topK 10 -T 6 > output_scidocs.log 2>&1 &

time ./calculate_scores -v data/amazon_vocab.json -d data/amazon_doc.json -o data/amazon_scores.jsonl -topK 10 -T 6 > output_amazon.log 2>&1 &

time ./calculate_scores -v data/french_news_vocab.json -d data/french_news_doc.json -o data/french_news_scores.jsonl -topK 10 -T 6 > output_french_news.log 2>&1 &

time ./calculate_scores -v data/merged_vocab.json -d data/merged_doc.json -o data/merged_scores.jsonl -topK 10 -T 6 > output_merged.log 2>&1 &
