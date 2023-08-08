import json
import re

import pandas as pd
from tqdm import tqdm

from src.rules import get_sentence_split_rules, get_vocab_removal_rules


class AmazonPreprocess:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.vocab_list = None

    def create_vocab_list(self):
        vocab_dict = {}
        paper_id_dict = {}
        with open(self.data_path, 'r') as f:
            for line in tqdm(f, desc="Processing Data"):
                data = json.loads(line)
                paper_id = data['document']
                for sentence in list(filter(None, re.split(get_sentence_split_rules(), data['text'].strip()))):
                    clean_vocabs = list(filter(None, re.sub(get_vocab_removal_rules(), ' ', sentence).strip().split()))
                    for vocab in clean_vocabs:
                        vocab_dict.setdefault(vocab, 0)
                        vocab_dict[vocab] += 1
                        paper_id_dict.setdefault(vocab, {})
                        paper_id_dict[vocab].setdefault(paper_id, 0)
                        paper_id_dict[vocab][paper_id] += 1
        vocab_dict_df = pd.DataFrame(list(vocab_dict.items()), columns=['vocab', 'count'])
        paper_id_dict_df = pd.DataFrame(list(paper_id_dict.items()), columns=['vocab', 'paper_ids'])
        return pd.merge(vocab_dict_df, paper_id_dict_df, on='vocab')

    def prepare_dataset(self):
        print("Creating Amazon Dataset ...")
        self.vocab_list = self.create_vocab_list()
        print()
        print(self.vocab_list)

    def save_vocab_list(self, save_path):
        print("\nSaving Vocab List ... ")
        self.vocab_list.to_pickle(save_path)


if __name__ == "__main__":
    amazon_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/amazon_data/dataset.json"

    # Initialize preproccess module #
    amazon_preproc = AmazonPreprocess(data_path=amazon_data_path)
    amazon_preproc.prepare_dataset()
    amazon_preproc.save_vocab_list(
        save_path="/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/amazon_data/amazon_vocab_cased.pkl")
