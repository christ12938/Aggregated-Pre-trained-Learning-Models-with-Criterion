import regex as re
from utils import clean_sentence, create_vocab_info_df, create_folder, create_doc_info_df, save_info


class FrenchNewsPreprocess():
    def __init__(self, french_news_data_path: str, id_prefix: str, sample=1):
        self.data = []
        self.vocab_info_df = None
        self.doc_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample
        self.open_file(french_news_data_path=french_news_data_path)


    def open_file(self, french_news_data_path: str):
        data_per_file = ''
        with open(french_news_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line == '\n':
                    self.data.append(data_per_file)
                    data_per_file = ''
                else:
                    data_per_file = data_per_file + line.strip() + ' '


    def preprocess(self):
        for idx, data in enumerate(self.data):
            self.data[idx] = clean_sentence(sentence=data.strip().lower(), regex_rules=r"(?!')[\p{P}\p{S}]")
        self.vocab_info_df = create_vocab_info_df(sentences_list=self.data,
                                                  id_prefix=self.id_prefix,
                                                  sample=self.sample)
        self.doc_info_df = create_doc_info_df(vocab_info_df=self.vocab_info_df)

    
    def save_info(self, vocab_info_save_path: str, doc_info_save_path: str):
        save_info("French News", vocab_info_save_path, doc_info_save_path, self.vocab_info_df, self.doc_info_df)
