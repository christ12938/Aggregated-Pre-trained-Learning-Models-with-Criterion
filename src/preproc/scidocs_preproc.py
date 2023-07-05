import json
import pandas as pd
import re
from tqdm import tqdm

from src.rules import get_sentence_split_rules, get_vocab_removal_rules


class SciDocsPreprocess:
    def __init__(self,
                 paper_metadata_mag_mesh_path,
                 paper_metadata_recomm_path,
                 paper_metadata_view_cite_read_path):
        self.scidocs_data = self.open_file(paper_metadata_mag_mesh_path, paper_metadata_recomm_path,
                                           paper_metadata_view_cite_read_path)
        self.split_sentences = None
        self.vocab_list = None

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
        for key, value in tqdm(self.scidocs_data.items(), desc="Splitting Dataset"):
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
        vocab_dict = {}
        sentences = list(self.split_sentences.loc[:, "sentences"])
        for sentence in tqdm(sentences, desc="Creating Vocabulary"):
            clean_vocabs = list(filter(None, re.sub(get_vocab_removal_rules(), ' ', sentence).strip().split()))
            for vocab in clean_vocabs:
                vocab_dict.setdefault(vocab, 0)
                vocab_dict[vocab] += 1
        return pd.DataFrame(list(vocab_dict.items()), columns=['vocab', 'count'])

    def prepare_dataset(self):
        print("Creating Scidocs Dataset ...")
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
