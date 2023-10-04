import pandas as pd
from utils import create_vocab_info_df, create_folder
from tqdm import tqdm


class AmazonPreprocess:
    def __init__(self, amazon_data_path: str, id_prefix: str, sample=1):
        self.data = []
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(amazon_data_path)


    def open_file(self, amazon_data_path: str):
        with open(amazon_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing Amazon Data"):
                self.data.append(line.strip())


    def preprocess(self):
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix, sample=self.sample)


    def save_vocab_info(self, save_path: str):
        print("\nSaving Amazon Vocab Info ... ")
        create_folder(path=save_path)
        self.vocab_info_df.to_pickle(save_path)

