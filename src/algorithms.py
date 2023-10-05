import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import create_folder
from metrics import calculate_score_per_seed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.functional import cosine_similarity



device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'



class CosineScore:
    def __init__(self, vocab_embeddings_path: str, seed_embeddings_path: str):
        self.vocab_embeddings_df = pd.read_pickle(vocab_embeddings_path)
        self.seed_embeddings_df = pd.read_pickle(seed_embeddings_path)
        self.result_dict = None

    def process_vocab_embeddings(self):
        vocab_embeddings_dict = self.vocab_embeddings_df.to_dict('records')
        seeds_words = list(self.seed_embeddings_df.loc[:, "seed"])
        seeds_embeddings_list = [torch.from_numpy(tensor).to(device) for tensor in
                                 list(self.seed_embeddings_df.loc[:, "embedding"])]
        self.result_dict = {key: [] for key in seeds_words}
       
        for row in tqdm(vocab_embeddings_dict, desc="Calculating Cosine Scores for Vocabs"):
            vocab = row['vocab']
            vocab_embedding = torch.from_numpy(row['embedding']).to(device)
           
            # calculate cosine scores
            cosine_scores = torch.Tensor([cosine_similarity(seed_embedding, vocab_embedding) for seed_embedding in seeds_embeddings_list])
            max_cosine_score = torch.max(cosine_scores)
            result_index = torch.argmax(cosine_scores)
            # append data to result
            self.result_dict[seeds_words[result_index]].append((vocab, max_cosine_score))

        # Sort scores based on cosine score
        for seed, words in tqdm(self.result_dict.items(), desc="Ranking Vocabs According to Cosine Scores"):
            sorted_vocabs = sorted(words, key=lambda x: x[1], reverse=True)
            self.result_dict[seed] = [vocab[0] for vocab in sorted_vocabs]


    def save_result(self, save_path: str):
        print("Saving Results ...")
        create_folder(path=save_path)
        pd.DataFrame(list(self.result_dict.items()), columns=['seed', 'words']).to_pickle(save_path)


class CombineCriteria:
    def __init__(self, vocab_info_path: str, categorized_words_paths: list, criteria: str, top_k: int):
        if criteria != 'ppmi' and criteria != 'npmi':
            raise Exception(f"No criteria named {criteria}")
        self.criteria = criteria
        self.top_k = top_k
        self.vocab_info_df = None
        self.categorized_words_dicts = None
        self.result_dict = None
        self.open_files(vocab_info_path=vocab_info_path, categorized_words_paths=categorized_words_paths)
    

    def open_files(self, vocab_info_path: str, categorized_words_paths: list):
        self.vocab_info_df = pd.read_pickle(vocab_info_path)
        self.categorized_words_dicts = [pd.read_pickle(path).set_index('seed')['words'].to_dict() for path in categorized_words_paths] 
        
        # Sanity Checks: TODO
    

    def combine_results(self):
        vocab_info_dict = self.vocab_info_df.set_index('vocab')['id'].to_dict()
        total_doc_count = len(set(element for entry in list(self.vocab_info_df.loc[:, 'id']) for element in entry))
        seeds = list(self.categorized_words_dicts[0].keys())
        self.result_dict = {}

        for seed in tqdm(seeds, desc=f"Calculating Highest {top_k} {criteria} for Each Seed"):
            seed_scores = np.zeros(len(self.categorized_words_dicts))
            for idx, result in enumerate(self.categorized_words_dicts):
                seed_scores[idx] = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, words=result[seed], 
                                                            top_k=top_k, total_doc_count=total_doc_count, measure=self.criteria)
            self.result_dict[seed] = self.categorized_words_dicts[np.argmax(seed_scores)][seed]


    def save_result(self, save_path: str):
        print("Saving Results ...")
        create_folder(path=save_path)
        pd.DataFrame(list(self.result_dict.items()), columns=['seed', 'words']).to_pickle(save_path)


if __name__ == "__main__":  

    # Seeds Embeddings
    scidocs_seed_embed_bert_base_save_path = "embeddings/scidocs_seed_tensors_bert_base_uncased.pkl"
    scidocs_seed_embed_scibert_save_path = "embeddings/scidocs_seed_tensors_scibert_uncased.pkl"
    scidocs_seed_embed_flaubert_save_path = "embeddings/scidocs_seed_tensors_flaubert_uncased.pkl"

    amazon_seed_embed_bert_base_save_path = "embeddings/amazon_seed_tensors_bert_base_uncased.pkl"
    amazon_seed_embed_scibert_save_path = "embeddings/amazon_seed_tensors_scibert_uncased.pkl"
    amazon_seed_embed_flaubert_save_path = "embeddings/amazon_seed_tensors_flaubert_uncased.pkl"

    french_seed_embed_bert_base_save_path = "embeddings/french_seed_tensors_bert_base_uncased.pkl"
    french_seed_embed_scibert_save_path = "embeddings/french_seed_tensors_scibert_uncased.pkl"
    french_seed_embed_flaubert_save_path = "embeddings/french_seed_tensors_flaubert_uncased.pkl"

    merged_seed_embed_bert_base_save_path = "embeddings/merged_seed_tensors_bert_base_uncased.pkl"
    merged_seed_embed_scibert_save_path = "embeddings/merged_seed_tensors_scibert_uncased.pkl"
    merged_seed_embed_flaubert_save_path = "embeddings/merged_seed_tensors_flaubert_uncased.pkl"

    
    # Vocab Embeddings
    scidocs_vocab_embed_bert_base_save_path = "embeddings/scidocs_vocab_tensors_bert_base_uncased.pkl"
    scidocs_vocab_embed_scibert_save_path = "embeddings/scidocs_vocab_tensors_scibert_uncased.pkl"
    scidocs_vocab_embed_flaubert_save_path = "embeddings/scidocs_vocab_tensors_flaubert_uncased.pkl"

    amazon_vocab_embed_bert_base_save_path = "embeddings/amazon_vocab_tensors_bert_base_uncased.pkl"
    amazon_vocab_embed_scibert_save_path = "embeddings/amazon_vocab_tensors_scibert_uncased.pkl"
    amazon_vocab_embed_flaubert_save_path = "embeddings/amazon_vocab_tensors_flaubert_uncased.pkl"

    french_vocab_embed_bert_base_save_path = "embeddings/french_vocab_tensors_bert_base_uncased.pkl"
    french_vocab_embed_scibert_save_path = "embeddings/french_vocab_tensors_scibert_uncased.pkl"
    french_vocab_embed_flaubert_save_path = "embeddings/french_vocab_tensors_flaubert_uncased.pkl"

    merged_vocab_embed_bert_base_save_path = "embeddings/merged_vocab_tensors_bert_base_uncased.pkl"
    merged_vocab_embed_scibert_save_path = "embeddings/merged_vocab_tensors_scibert_uncased.pkl"
    merged_vocab_embed_flaubert_save_path = "embeddings/merged_vocab_tensors_flaubert_uncased.pkl"

   
    # Result paths
    scidocs_bert_base_result_save_path = "result_data/scidocs_result_bert_base_uncased.pkl"
    scidocs_scibert_result_save_path = "result_data/scidocs_result_scibert_uncased.pkl"
    scidocs_flaubert_result_save_path = "result_data/scidocs_result_flaubert_uncased.pkl"
    scidocs_combined_npmi_result_save_path = "result_data/scidocs_result_combined_npmi.pkl"

    amazon_bert_base_result_save_path = "result_data/amazon_result_bert_base_uncased.pkl"
    amazon_scibert_result_save_path = "result_data/amazon_result_scibert_uncased.pkl"
    amazon_flaubert_result_save_path = "result_data/amazon_result_flaubert_uncased.pkl"
    amazon_combined_npmi_result_save_path = "result_data/amazon_result_combined_npmi.pkl"

    french_bert_base_result_save_path = "result_data/french_result_bert_base_uncased.pkl"
    french_scibert_result_save_path = "result_data/french_result_scibert_uncased.pkl"
    french_flaubert_result_save_path = "result_data/french_result_flaubert_uncased.pkl"
    french_combined_npmi_result_save_path = "result_data/french_result_combined_npmi.pkl"

    merged_bert_base_result_save_path = "result_data/merged_result_bert_base_uncased.pkl"
    merged_scibert_result_save_path = "result_data/merged_result_scibert_uncased.pkl"
    merged_flaubert_result_save_path = "result_data/merged_result_flaubert_uncased.pkl"
    merged_combined_npmi_result_save_path = "result_data/merged_result_combined_npmi.pkl"


    # Scidocs
    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_bert_base_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_bert_base_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_bert_base_result_save_path)
    del scidocs_max_cosine

    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_scibert_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_scibert_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_scibert_result_save_path)
    del scidocs_max_cosine
    
    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_flaubert_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_flaubert_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_flaubert_result_save_path)
    del scidocs_max_cosine
    
    # Amazon
    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_bert_base_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_bert_base_result_save_path)
    del amazon_max_cosine

    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_scibert_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_scibert_result_save_path)
    del amazon_max_cosine

    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_flaubert_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_flaubert_result_save_path)
    del amazon_max_cosine
    
    # XLSum FR
    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=french_seed_embed_bert_base_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_bert_base_result_save_path)
    del french_max_cosine
    
    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=french_seed_embed_scibert_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_scibert_result_save_path)
    del french_max_cosine

    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=french_seed_embed_flaubert_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_flaubert_result_save_path)
    del french_max_cosine

    # Merged
    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=merged_seed_embed_bert_base_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_bert_base_result_save_path)
    del merged_max_cosine

    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=merged_seed_embed_scibert_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_scibert_result_save_path)
    del merged_max_cosine

    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=merged_seed_embed_flaubert_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_flaubert_result_save_path)
    del merged_max_cosine

    # Combined NPMI method
    scidocs_vocab_path = "data/scidocs_vocab.pkl"
    amazon_vocab_path = "data/amazon_vocab.pkl"
    french_vocab_path = "data/xlsum_fr_vocab.pkl"
    merged_vocab_path = "data/merged_vocab.pkl"

    scidocs_words_paths = [scidocs_bert_base_result_save_path, 
                           scidocs_scibert_result_save_path,
                           scidocs_flaubert_result_save_path]
    amazon_words_paths = [amazon_bert_base_result_save_path, 
                          amazon_scibert_result_save_path, 
                          amazon_flaubert_result_save_path] 
    french_words_paths = [french_bert_base_result_save_path,
                          french_scibert_result_save_path, 
                          french_flaubert_result_save_path]
    merged_words_paths = [merged_bert_base_result_save_path, 
                          merged_scibert_result_save_path, 
                          merged_flaubert_result_save_path]

    criteria = 'npmi'
    top_k = 20

    scidocs_combined_npmi = CombineCriteria(vocab_info_path=scidocs_vocab_path,
                                            categorized_words_paths=scidocs_words_paths, 
                                            criteria=criteria, 
                                            top_k=top_k)
    scidocs_combined_npmi.combine_results()
    scidocs_combined_npmi.save_result(save_path=scidocs_combined_npmi_result_save_path)
    del scidocs_combined_npmi

    amazon_combined_npmi = CombineCriteria(vocab_info_path=amazon_vocab_path,
                                            categorized_words_paths=amazon_words_paths,
                                            criteria=criteria, 
                                            top_k=top_k)
    amazon_combined_npmi.combine_results()
    amazon_combined_npmi.save_result(save_path=amazon_combined_npmi_result_save_path)
    del amazon_combined_npmi

    french_combined_npmi = CombineCriteria(vocab_info_path=french_vocab_path,
                                            categorized_words_paths=french_words_paths, 
                                            criteria=criteria, 
                                            top_k=top_k)
    french_combined_npmi.combine_results()
    french_combined_npmi.save_result(save_path=french_combined_npmi_result_save_path)
    del french_combined_npmi

    merged_combined_npmi = CombineCriteria(vocab_info_path=merged_vocab_path,
                                            categorized_words_paths=merged_words_paths, 
                                            criteria=criteria, 
                                            top_k=top_k)
    merged_combined_npmi.combine_results()
    merged_combined_npmi.save_result(save_path=merged_combined_npmi_result_save_path)
    del merged_combined_npmi
