import re

import pandas as pd
from tqdm import tqdm

from src.rules import get_sentence_split_rules, get_vocab_removal_rules


class LegalDataPreproc:
    def __init__(self, data_path: str):
        self.legal_data = pd.read_csv(data_path, na_filter=False)
        self.result_df = None

    def process_dataset(self):
        sentence_list = (self.legal_data['case_title'] + ' ' + self.legal_data['case_text']).to_list()
        result_dict = {}
        for paper_id, sentences in enumerate(tqdm(sentence_list, desc="Processing Dataset")):
            sentence = list(filter(None, re.split(get_sentence_split_rules(), sentences.strip().lower())))
            for clean_sentence in sentence:
                clean_vocabs = list(
                    filter(None, re.sub(get_vocab_removal_rules(), ' ', clean_sentence).strip().split()))
                for vocab in clean_vocabs:
                    result_dict.setdefault(vocab, set()).add(f'LEGAL{paper_id}')
        self.result_df = pd.DataFrame(list(result_dict.items()), columns=['vocab', 'paper_ids'])

    def save_vocab_list(self, save_path: str):
        print("\nSaving Vocab List ... ")
        print(self.result_df)
        self.result_df.to_pickle(save_path)


if __name__ == "__main__":
    legald_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/legal_data/legal_text_classification.csv"
    legal_data_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/legal_vocab.pkl"

    legal_data_preproc = LegalDataPreproc(data_path=legald_data_path)
    legal_data_preproc.process_dataset()
    legal_data_preproc.save_vocab_list(save_path=legal_data_save_path)
