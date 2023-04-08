import json
import re

import pandas as pd


def get_seeds():
    seeds = [
        'cardiovascular diseases',
        'chronic kidney disease',
        'chronic respiratory diseases',
        'diabetes mellitus',
        'digestive diseases',
        'hiv/aids',
        'hepatitis a/b/c/e',
        'mental disorders',
        'musculoskeletal disorders',
        'neoplasms (cancer)',
        'neurological disorders'
    ]
    return seeds


def check_vocab_seeds(filepath: str):
    vocabs_df = pd.read_csv(filepath)
    seeds = get_seeds()
    in_vocab_seeds = []
    vocab_list = []
    count = 1

    for key, value in scidocs_dict.items():
        if value['abstract'] is None:
            value['abstract'] = ''
        if value['title'] is None:
            value['title'] = ''
        value['abstract'] = value['abstract'].lower()
        value['title'] = value['title'].lower()

        print('[' + str(count) + '/' + str(len(scidocs_dict)) + ']' + " ... Unpacking Contents", end='\r')
        for seed in seeds:
            if seed not in in_vocab_seeds and (seed in value['abstract'] or seed in value['title']):
                in_vocab_seeds.append(seed)
                break
        vocab_list.extend(re.split(r'[,.\s]', value['abstract']))
        vocab_list.extend(re.split(r'[,.\s]', value['title']))
        count += 1

    print()
    print("Removing Duplicates ...")
    vocab_list = list(set(vocab_list))

    print(len(vocab_list))
    print(in_vocab_seeds)
    # return vocab_list


if __name__ == "__main__":
    vocab_128 = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_vocab_128.csv"
    vocab_256 = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_vocab_256.csv"
    vocab_512 = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_vocab_512.csv"

