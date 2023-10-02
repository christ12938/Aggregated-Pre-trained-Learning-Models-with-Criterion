import json
import pandas as pd
from tqdm import tqdm
from utils import clean_sentence, create_vocab_info_df


class French_XLSum_2_0_Preproc():
    def __init__(self, file_path: str, id_prefix: str):
        self.data = []
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
               self.data.append(json.loads(line)) 


    def preprocess(self):
        data_dict = {}
        for entry in tqdm(self.data, desc='Preprocessing Data'):
            combined_text = (entry['title'] + ' ' + entry['summary'] + ' ' + entry['text']).lower()
            cleaned_sentence = clean_sentence(sentence=combined_text, regex_rules=r"(?!')[\p{P}\p{S}]")
            if entry['id'] in data_dict:
                data_dict[entry['id']] = data_dict[entry['id']] + ' ' + cleaned_sentence
            else:
                data_dict[entry['id']] = cleaned_sentence
        self.vocab_info_df = create_vocab_info_df(sentences_list=list(data_dict.values()), id_prefix=self.id_prefix)

    def save_vocab_info(self, save_path: str):
        self.vocab_info_df.to_pickle(save_path)

if __name__ == '__main__': 
    file_path = "french_train.jsonl"
    vocab_info_df_save_path = "french_vocab.pkl"
    id_prefix = "FR"

    french_data_preproc = French_XLSum_2_0_Preproc(file_path=file_path, id_prefix=id_prefix)
    french_data_preproc.preprocess()
    french_data_preproc.save_vocab_info(save_path=vocab_info_df_save_path) 

