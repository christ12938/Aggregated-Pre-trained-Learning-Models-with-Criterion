import pandas as pd
from utils import create_folder


class MergedPreprocess():
    def __init__(self, vocab_info_df_list: list):
        self.vocab_info_df_list = vocab_info_df_list
        self.vocab_info_df = None


    @staticmethod
    def __combine_sets(series):
        combined_set = set()
        for e in series:
            combined_set = combined_set.union(e)
        return combined_set


    def preprocess(self):
        # Assume sentences are already cleaned
        combined_df = pd.concat(self.vocab_info_df_list)
        self.vocab_info_df = combined_df.groupby('vocab').agg({'id': self.__combine_sets}).reset_index()


    def save_vocab_info(self, save_path: str):
        print("\nSaving Merged Vocab Info ... ")
        create_folder(path=save_path)
        self.vocab_info_df.to_pickle(save_path)

