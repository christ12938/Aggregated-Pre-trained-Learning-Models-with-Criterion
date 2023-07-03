import json
import pandas as pd
import re
import random
from transformers import BertTokenizer, logging
from collections import Counter

from src.rules import get_sentence_split_rules, get_vocab_removal_rules


class SciDocsPreprocess:
    def __init__(self,
                 paper_metadata_mag_mesh_path,
                 paper_metadata_recomm_path,
                 paper_metadata_view_cite_read_path,
                 vocab_path,
                 keep_no_nsp,
                 max_len):
        self.scidocs_data = self.open_file(paper_metadata_mag_mesh_path, paper_metadata_recomm_path,
                                           paper_metadata_view_cite_read_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.keep_no_nsp = keep_no_nsp
        self.scidocs_bert = None
        self.vocab_list = None
        self.vocab_path = vocab_path
        logging.set_verbosity_error()

    def open_file(self, paper_metadata_mag_mesh_path, paper_metadata_recomm_path, paper_metadata_view_cite_read_path):
        paper_metadata_mag_mesh = paper_metadata_recomm = paper_metadata_view_cite_read = None

        with open(paper_metadata_mag_mesh_path, 'r') as f1:
            paper_metadata_mag_mesh = json.load(f1)

        with open(paper_metadata_recomm_path, 'r') as f2:
            paper_metadata_recomm = json.load(f2)

        with open(paper_metadata_view_cite_read_path, 'r') as f3:
            paper_metadata_view_cite_read = json.load(f3)

        if paper_metadata_mag_mesh is None or paper_metadata_recomm is None or paper_metadata_view_cite_read is None:
            raise Exception("SciDocs Data is not Accessible.")

        return paper_metadata_mag_mesh | paper_metadata_recomm | paper_metadata_view_cite_read

    # TODO: Refactor line 50 ~ 61 #
    def split_dataset(self):
        print("Splitting Dataset ... ")
        split_dict = {}
        count = 0
        for key, value in self.scidocs_data.items():
            if value['abstract'] is None:
                value['abstract'] = ''
            if value['title'] is None:
                value['title'] = ''
            abstracts = list(filter(None, re.split(get_sentence_split_rules(), value['abstract'].strip().lower())))
            titles = list(filter(None, re.split(get_sentence_split_rules(), value['title'].strip().lower())))
            for i in range(0, len(abstracts)):
                if i == len(abstracts) - 1:
                    split_dict[count] = {"sentence": abstracts[i].strip(), "next_sentence_id": None}
                else:
                    split_dict[count] = {"sentence": abstracts[i].strip(), "next_sentence_id": count + 1}
                count += 1
            for i in range(0, len(titles)):
                if i == len(titles) - 1:
                    split_dict[count] = {"sentence": titles[i].strip(), "next_sentence_id": None}
                else:
                    split_dict[count] = {"sentence": titles[i].strip(), "next_sentence_id": count + 1}
                count += 1
        return split_dict

    # TODO: Cater for NSP data ratio, 10% data loss #
    # TODO: Remove commented lines #
    def create_dataset_for_bert(self, split_dict):
        bert_dict = {"sentence_1": [], "sentence_2": [], "nsp": []}
        nsp_flag = True
        for key, value in split_dict.items():
            print("Creating Dataset ... [{curr} / {total}]".format(curr=str(key + 1), total=str(len(split_dict))),
                  end='\r')
            bert_dict["sentence_1"].append(value["sentence"])
            bert_dict["nsp"].append(1) if nsp_flag else bert_dict["nsp"].append(0)

            if self.keep_no_nsp:
                bert_dict["sentence_2"].append(None)
            elif nsp_flag is True:
                if value["next_sentence_id"] is None:
                    bert_dict["sentence_1"].pop()
                    bert_dict["nsp"].pop()
                else:
                    bert_dict["sentence_2"].append(split_dict[value["next_sentence_id"]]["sentence"])
                    #sentence_1_masked, sentence_1_masking_ids = self.mask_sentence(value["sentence"])
                    #sentence_2_masked, sentence_2_masking_ids = self.mask_sentence(splitted_dict[value["next_sentence_id"]]["sentence"])
                    #bert_dict["sentence_1_masked"].append(sentence_1_masked)
                    #bert_dict["sentence_1_masking_ids"].append(sentence_1_masking_ids)
                    #bert_dict["sentence_2_masked"].append(sentence_2_masked)
                    #bert_dict["sentence_2_masking_ids"].append(sentence_2_masking_ids)
            else:
                random_index = 0
                while True:
                    random_index = random.randint(0, len(split_dict) - 1)
                    if random_index != key and random_index != key + 1:
                        break
                #sentence_1_masked, sentence_1_masking_ids = self.mask_sentence(value["sentence"])
                #sentence_2_masked, sentence_2_masking_ids = self.mask_sentence(splitted_dict[random_index]["sentence"])
                #bert_dict["sentence_1_masked"].append(sentence_1_masked)
                #bert_dict["sentence_1_masking_ids"].append(sentence_1_masking_ids)
                bert_dict["sentence_2"].append(split_dict[random_index]["sentence"])
                #bert_dict["sentence_2_masked"].append(sentence_2_masked)
                #bert_dict["sentence_2_masking_ids"].append(sentence_2_masking_ids)
            nsp_flag = not nsp_flag
        return pd.DataFrame(data=bert_dict)

    def mask_sentence(self, sentence):
        masking_rate = 0.15
        rand_rate = masking_rate * 0.1
        masking_ids = []
        sentence_dict = self.tokenizer(sentence, add_special_tokens=False)
        for i in range(0, len(sentence_dict['input_ids'])):
            if random.random() < masking_rate:
                masking_ids.append(i)
                new_id = 103
                #if random.random() < rand_rate:
                    #new_id = random.randint(999, self.tokenizer.vocab_size - 1)
                #else:
                    #new_id = 103
                sentence_dict['input_ids'][i] = new_id
        return self.tokenizer.decode(sentence_dict['input_ids']), masking_ids

    def create_vocab_list(self):
        sentence_1 = list(self.scidocs_bert.loc[:, "sentence_1"])
        sentence_2 = list(self.scidocs_bert.loc[:, "sentence_2"])
        combined_sentence = []
        for i in range(len(sentence_1)):
            if self.keep_no_nsp:
                new_sentence_1 = re.sub(r'[^\w\s\'-]|\d', ' ', sentence_1[i])
                combined_sentence.append(new_sentence_1)
            else:
                combined_sentence.append([sentence_1[i], sentence_2[i]])

        print("\nTokenizing Dataset ... ")
        tokenized = self.tokenizer(combined_sentence, add_special_tokens=False, truncation=True,
                                   max_length=self.max_len - 3)
        vocab_list = []
        print("Decoding Dataset ... ")
        for idx, i in enumerate(tokenized['input_ids']):
            print("Decoding Dataset ... [{curr} / {total}]".format(curr=str(idx + 1), total=str(len(tokenized['input_ids']))),
                  end='\r')
            vocab_list.extend(list(filter(None, self.tokenizer.decode(i).strip().split(" "))))
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
        self.scidocs_bert = self.create_dataset_for_bert(self.split_dataset())
        self.vocab_list = self.create_vocab_list()
        print()
        print(self.vocab_list)
    
    def save_scidocs_data(self, save_path):
        print("\nSaving Dataset ... ")
        self.scidocs_bert.to_csv(save_path, index=False)

    def save_vocab_list(self, save_path):
        print("\nSaving Vocab List ... ")
        self.vocab_list.to_csv(save_path, index=False)
