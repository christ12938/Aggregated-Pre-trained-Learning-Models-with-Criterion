import regex as re
import pandas as pd
from utils import create_vocab_info_df, create_folder, create_doc_info_df, save_info, clean_sentence
from tqdm import tqdm


class AmazonPreprocess:
    def __init__(self, amazon_data_path: str, id_prefix: str, sample=1):
        self.data = []
        self.vocab_info_df = None
        self.doc_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(amazon_data_path)


    def open_file(self, amazon_data_path: str):
        with open(amazon_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing Amazon Data"):
                self.data.append(line.strip())


    def preprocess(self):
        for idx, data in enumerate(self.data):
            self.data[idx] = clean_sentence(sentence=data.strip().lower(), regex_rules=r'^$')
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix, sample=self.sample)
        self.doc_info_df = create_doc_info_df(vocab_info_df=self.vocab_info_df)

    def save_info(self, vocab_info_save_path: str, doc_info_save_path: str):
        save_info("Amazon", vocab_info_save_path, doc_info_save_path, self.vocab_info_df, self.doc_info_df)
