import pandas as pd
from utils import create_vocab_info_df


class AmazonPreprocess:
    def __init__(self, amazon_data_path: str, id_prefix: str):
        self.data = []
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        open_file(amazon_data_path)


    @staticmethod
    def open_file(amazon_data_path: str):
        with open(amazon_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line.strip())

    def preprocess(self):
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix)


    def save_vocab_info(self, save_path: str):
        print("\nSaving Amazon Vocab Info ... ")
        self.vocab_info_df.to_pickle(save_path)

