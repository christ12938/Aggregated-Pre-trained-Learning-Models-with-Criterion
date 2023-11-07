import regex as re
import pandas as pd
from utils import create_vocab_info_df, create_folder, create_doc_info_df, save_info, clean_sentence, save_seetopic_data
from tqdm import tqdm
import numpy as np


class SeetopicScidocsPreprocess:
    def __init__(self, scidocs_data_path: str, id_prefix: str, sample=1):
        self.data = []
        self.vocab_info_df = None
        self.doc_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(scidocs_data_path)


    def open_file(self, scidocs_data_path: str):
        with open(scidocs_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing Scidocs Data"):
                self.data.append(line.strip())

    def __add_idf(self, total_doc_count: int):
        self.vocab_info_df['idf'] = np.log(total_doc_count / self.vocab_info_df['id'].apply(lambda x: len(x)))
        min_idf = self.vocab_info_df['idf'].min()
        max_idf = self.vocab_info_df['idf'].max()

        # Apply standardization: (x - min) / (max - min)
        self.vocab_info_df['standardized_idf'] = (self.vocab_info_df['idf'] - min_idf) / (max_idf - min_idf)

    def preprocess(self):
        for idx, data in enumerate(self.data):
            self.data[idx] = clean_sentence(sentence=data.strip().lower(), regex_rules=r'^$')
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix, sample=self.sample)
        self.doc_info_df = create_doc_info_df(vocab_info_df=self.vocab_info_df)
        self.__add_idf(total_doc_count=len(self.doc_info_df['id']))

    def save_info(self, vocab_info_save_path: str, doc_info_save_path: str, seetopic_data_save_path: str):
        save_info("Scidocs", vocab_info_save_path, doc_info_save_path, self.vocab_info_df, self.doc_info_df)
        save_seetopic_data(data=self.data, path=seetopic_data_save_path)
