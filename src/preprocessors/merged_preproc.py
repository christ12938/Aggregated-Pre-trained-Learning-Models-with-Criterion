import pandas as pd
from utils import create_vocab_info_df


class MergedPreprocess():
    def __init__(self, sentences: list, id_prefix):
        self.sentences = sentences
        self.vocab_info_df = None
        self.id_prefix


    def preprocess(self):
        # Assume sentences are already cleaned
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.sentences, id_prefix=self.id_prefix)


    def save_vocab_info(self, save_path: str):
        print("\nSaving Merged Vocab Info ... ")
        self.vocab_info_df.to_pickle(save_path)

