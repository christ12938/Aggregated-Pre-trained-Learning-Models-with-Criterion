import pandas as pd
from utils import create_vocab_info_df, create_folder, create_doc_info_df, save_info
from tqdm import tqdm


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


    def preprocess(self):
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix, sample=self.sample)
        self.doc_info_df = create_doc_info_df(vocab_info_df=self.vocab_info_df)


    def save_info(self, vocab_info_save_path: str, doc_info_save_path: str):
        save_info("Scidocs", vocab_info_save_path, doc_info_save_path, self.vocab_info_df, self.doc_info_df)
