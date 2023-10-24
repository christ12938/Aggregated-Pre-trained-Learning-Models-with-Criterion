import sys
import pandas as pd
from seeds import get_scidocs_seeds, get_amazon_seeds, get_french_seeds

print()
decision_df = pd.read_pickle(sys.argv[1])
records = decision_df.to_records(index=False)
count = 0
for record in records:
    seed = record['seed']
    decision = record['decision']
    if seed in get_scidocs_seeds() and decision != 'SciBert':
        count += 1
        print(f'Misclassified Seed: {seed}, Current: {decision}, Correct: SciBert')
    elif seed in get_amazon_seeds() and decision != 'Bert Base':
        count += 1
        print(f'Misclassified Seed: {seed}, Current: {decision}, Correct: Bert Base')
    elif seed in get_french_seeds() and decision != 'FlauBert':
        count += 1
        print(f'Misclassified Seed: {seed}, Current: {decision}, Correct: FlauBert')

print(f'Total Misclassified Seeds: {count}')
