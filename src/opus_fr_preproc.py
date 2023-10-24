import pandas as pd
from utils import create_vocab_info_df, create_folder, clean_sentence
from tqdm import tqdm
import random


class Opus_Fr_Preprocess:
    def __init__(self, opus_fr_data_path: str, id_prefix: str, sample=1):
        self.data = []
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(opus_fr_data_path)


    def open_file(self, opus_data_path: str):
        data_per_file = ""
        with open(opus_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    self.data.append(clean_sentence(sentence=data_per_file.strip().lower(), regex_rules=r"(?!')[\p{P}\p{S}]"))
                    data_per_file = ""
                else:
                    data_per_file = data_per_file + line.strip() + ' '


    def preprocess(self):
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data, id_prefix=self.id_prefix, sample=self.sample)


    def save_vocab_info(self, save_path: str):
        print("\nSaving Opus Fr Vocab Info ... ")
        create_folder(path=save_path)
        self.vocab_info_df.to_pickle(save_path)


if __name__ == '__main__':

    opus_fr_data_path = "../opus_fr_data/news_commentary_fr.txt"
    opus_fr_vocab_save_path = "data/xlsum_fr_vocab.pkl"

    opus_fr_preproc = Opus_Fr_Preprocess(opus_fr_data_path=opus_fr_data_path, id_prefix='NEWS_FR')
    opus_fr_preproc.preprocess()
    opus_fr_preproc.save_vocab_info(save_path=opus_fr_vocab_save_path)

