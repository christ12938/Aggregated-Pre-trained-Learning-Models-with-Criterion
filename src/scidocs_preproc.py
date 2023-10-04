import json
import pandas as pd
from tqdm import tqdm
from utils import clean_sentence, create_vocab_info_df, create_folder


class SciDocsPreprocess:
    def __init__(self,
            paper_metadata_mag_mesh_path: str,
            paper_metadata_recomm_path: str,
            paper_metadata_view_cite_read_path: str, 
            id_prefix: str,
            sample=1):
        
        self.scidocs_data = self.open_file(paper_metadata_mag_mesh_path, paper_metadata_recomm_path,
                                           paper_metadata_view_cite_read_path)
        
        self.data_dict = {}
        self.vocab_info_df = None
        self.id_prefix = id_prefix
        self.sample = sample


    @staticmethod
    def open_file(paper_metadata_mag_mesh_path: str, paper_metadata_recomm_path: str, paper_metadata_view_cite_read_path: str):

        with open(paper_metadata_mag_mesh_path, 'r', encoding='utf-8') as f1:
            paper_metadata_mag_mesh = json.load(f1)

        with open(paper_metadata_recomm_path, 'r', encoding='utf-8') as f2:
            paper_metadata_recomm = json.load(f2)

        with open(paper_metadata_view_cite_read_path, 'r', encoding='utf-8') as f3:
            paper_metadata_view_cite_read = json.load(f3)

        if paper_metadata_mag_mesh is None or paper_metadata_recomm is None or paper_metadata_view_cite_read is None:
            raise Exception("SciDocs Data is not Accessible.")

        return paper_metadata_mag_mesh | paper_metadata_recomm | paper_metadata_view_cite_read


    # Split dataset on punctuations
    def preprocess(self):
        for key, value in tqdm(self.scidocs_data.items(), desc="Processing Scidocs Dataset"):
            if value['abstract'] is None:
                value['abstract'] = ''
            if value['title'] is None:
                value['title'] = ''
            
            combined_text = (value['abstract'] + ' ' + value['title']).lower()
            cleaned_sentence = clean_sentence(sentence=combined_text, regex_rules=r'[^\w\s-]')
            if key in self.data_dict:
                self.data_dict[key] = self.data_dict[key] + ' ' + cleaned_sentence
            else:
                self.data_dict[key] = cleaned_sentence
        
        self.vocab_info_df = create_vocab_info_df(sentences_list=list(self.data_dict.values()), id_prefix=self.id_prefix, sample=self.sample) 


    def save_vocab_info(self, save_path: str):
        print("\nSaving Scidocs Vocab Info ... ")
        create_folder(path=save_path)
        self.vocab_info_df.to_pickle(save_path)



if __name__ == "__main__":

    data_1_path = '../../scidocs_data/paper_metadata_mag_mesh.json'
    data_2_path = '../../scidocs_data/paper_metadata_recomm.json'
    data_3_path = '../../scidocs_data/paper_metadata_view_cite_read.json'

    # Initialize preproccess module #
    scidocs_preproc = SciDocsPreprocess(data_1_path, data_2_path, data_3_path)
    scidocs_preproc.prepare_dataset()
    scidocs_preproc.save_scidocs_data(save_path="../data/scidocs_data/scidocs_dataset_sentence_split_raw.pkl")
    scidocs_preproc.save_vocab_list(save_path="../data/scidocs_data/scidocs_vocab_cased.pkl")
    
