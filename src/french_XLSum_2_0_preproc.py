import json
import pandas as pd
from tqdm import tqdm
from utils import clean_sentence, create_vocab_info_df, create_folder


class French_XLSum_2_0_Preprocess():
    def __init__(self, french_train_path: str, french_test_path: str, french_val_path: str, id_prefix: str, sample=1):
        self.data = []
        self.data_dict = {}
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(french_train_path, french_test_path, french_val_path)


    def open_file(self, french_train_path: str, french_test_path: str, french_val_path: str):
        
        with open(french_train_path, 'r', encoding='utf-8') as f1:
            for line in f1:
               self.data.append(json.loads(line)) 

        with open(french_test_path, 'r', encoding='utf-8') as f2:
            for line in f2:
               self.data.append(json.loads(line)) 

        with open(french_val_path, 'r', encoding='utf-8') as f3:
            for line in f3:
               self.data.append(json.loads(line)) 


    def preprocess(self):
        for entry in tqdm(self.data, desc='Processing XLSUM FR Data'):
            combined_text = (entry['title'] + ' ' + entry['summary'] + ' ' + entry['text']).lower()
            cleaned_sentence = clean_sentence(sentence=combined_text, regex_rules=r"(?!')[\p{P}\p{S}]")
            if entry['id'] in self.data_dict:
                self.data_dict[entry['id']] = self.data_dict[entry['id']] + ' ' + cleaned_sentence
            else:
                self.data_dict[entry['id']] = cleaned_sentence
        self.vocab_info_df = create_vocab_info_df(sentences_list=list(self.data_dict.values()), id_prefix=self.id_prefix, sample=self.sample)


    def save_vocab_info(self, save_path: str):
        print("\nSaving XLSum FR Vocab Info ... ")
        create_folder(path=save_path)
        self.vocab_info_df.to_pickle(save_path)


if __name__ == '__main__': 
    file_path = "french_train.jsonl"
    vocab_info_df_save_path = "french_vocab.pkl"
    id_prefix = "FR"

    french_data_preproc = French_XLSum_2_0_Preproccess(file_path=file_path, id_prefix=id_prefix)
    french_data_preproc.preprocess()
    french_data_preproc.save_vocab_info(save_path=vocab_info_df_save_path) 

