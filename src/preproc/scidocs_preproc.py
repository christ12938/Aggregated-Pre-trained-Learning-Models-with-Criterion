import json
import pandas as pd
import re
import torch
from transformers import BertTokenizer, logging
from collections import Counter

from src.rules import get_sentence_split_rules, get_vocab_removal_rules


class SciDocsPreprocess:
    def __init__(self,
                 paper_metadata_mag_mesh_path,
                 paper_metadata_recomm_path,
                 paper_metadata_view_cite_read_path,
                 vocab_path,
                 max_len):
        self.scidocs_data = self.open_file(paper_metadata_mag_mesh_path, paper_metadata_recomm_path,
                                           paper_metadata_view_cite_read_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.split_sentences = None
        self.vocab_list = None
        self.vocab_path = vocab_path
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
        logging.set_verbosity_error()

    @staticmethod
    def open_file(paper_metadata_mag_mesh_path, paper_metadata_recomm_path, paper_metadata_view_cite_read_path):

        with open(paper_metadata_mag_mesh_path, 'r') as f1:
            paper_metadata_mag_mesh = json.load(f1)

        with open(paper_metadata_recomm_path, 'r') as f2:
            paper_metadata_recomm = json.load(f2)

        with open(paper_metadata_view_cite_read_path, 'r') as f3:
            paper_metadata_view_cite_read = json.load(f3)

        if paper_metadata_mag_mesh is None or paper_metadata_recomm is None or paper_metadata_view_cite_read is None:
            raise Exception("SciDocs Data is not Accessible.")

        return paper_metadata_mag_mesh | paper_metadata_recomm | paper_metadata_view_cite_read

    # Split dataset on punctuations
    def split_dataset(self):
        sentence_list = []
        for count, (key, value) in enumerate(self.scidocs_data.items()):
            print("Splitting Dataset ... [{curr} / {total}]".format(curr=str(count + 1),
                                                                    total=str(len(self.scidocs_data))),
                  end='\r')
            import sys
            sys.stdout.flush()
            if value['abstract'] is None:
                value['abstract'] = ''
            if value['title'] is None:
                value['title'] = ''
            abstracts = list(filter(None, re.split(get_sentence_split_rules(), value['abstract'].strip().lower())))
            titles = list(filter(None, re.split(get_sentence_split_rules(), value['title'].strip().lower())))
            for abstract in abstracts:
                sentence_list.append(abstract.strip())
            for title in titles:
                sentence_list.append(title.strip())
        return pd.DataFrame(sentence_list, columns=['sentences'])

    def create_vocab_list(self):
        vocab_list = []
        tokenized_ids = []
        sentences = list(self.split_sentences.loc[:, "sentences"])
        for i, sentence in enumerate(sentences):
            print("Tokenizing Dataset ... [{curr} / {total}]".format(curr=str(i + 1),
                                                                     total=str(len(sentences))),
                  end='\r')
            tokenized = self.tokenizer([re.sub(r'[^\w\s\'-]|\d', ' ', sentence)], add_special_tokens=False,
                                       truncation=True, return_tensors="pt",
                                       max_length=self.max_len - 3).to(self.device)
            tokenized_ids.append(tokenized['input_ids'][0])

        for idx, id in enumerate(tokenized_ids):
            print("Decoding Dataset ... [{curr} / {total}]".format(curr=str(idx + 1),
                                                                   total=str(len(tokenized_ids))),
                  end='\r')
            vocab_list.extend(list(filter(None, self.tokenizer.decode(id).strip().split())))
        vocab_dict = Counter(vocab_list)
        bert_vocab_list = []
        f = open(self.vocab_path, "r", encoding="utf-8")
        for x in f:
            bert_vocab_list.append(x.strip())
        count = 0
        total_keys = len(list(vocab_dict.keys()))
        for key in list(vocab_dict.keys()):
            print("Removing Duplicate Vocabs ... [{curr} / {total}]"
                  .format(curr=str(count + 1), total=str(total_keys)), end="\r")
            if key in bert_vocab_list:
                del vocab_dict[key]
                bert_vocab_list.remove(key)
            count += 1
        vocab_list = vocab_dict.most_common()
        return pd.DataFrame(data={"vocabs": [x[0] for x in vocab_list], "count": [x[1] for x in vocab_list]})

    def prepare_dataset(self):
        print("Creating Scidocs Dataset for Length {}".format(str(self.max_len)))
        self.split_sentences = self.split_dataset()
        self.vocab_list = self.create_vocab_list()
        print()
        print(self.vocab_list)

    def save_scidocs_data(self, save_path):
        print("\nSaving Dataset ... ")
        self.split_sentences.to_csv(save_path, index=False)

    def save_vocab_list(self, save_path):
        print("\nSaving Vocab List ... ")
        self.vocab_list.to_csv(save_path, index=False)
