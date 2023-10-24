import pandas as pd
from utils import create_folder
from utils import create_folder, create_doc_info_df, save_info

class MergedPreprocess():
    def __init__(self, vocab_info_df_list: list):
        self.vocab_info_df_list = vocab_info_df_list
        self.vocab_info_df = None
        self.doc_info_df = None

    @staticmethod
    def __combine_dicts(series):
        combined_dict = dict()
        for e in series:
            combined_dict.update(e)
        return combined_dict

    def __add_vocab_count(self):
        self.vocab_info_df['count'] = self.vocab_info_df['id'].apply(lambda x: sum(x.values()))

    def preprocess(self):
        # Assume sentences are already cleaned
        combined_df = pd.concat(self.vocab_info_df_list)
        self.vocab_info_df = combined_df.groupby('vocab').agg({'id': self.__combine_dicts}).reset_index()
        self.__add_vocab_count()
        self.doc_info_df = create_doc_info_df(vocab_info_df=self.vocab_info_df)

    def save_info(self, vocab_info_save_path: str, doc_info_save_path: str):
        save_info("Merged", vocab_info_save_path, doc_info_save_path, self.vocab_info_df, self.doc_info_df)
