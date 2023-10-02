import math

import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import permutations, combinations
from sklearn.metrics.pairwise import cosine_similarity


def calculate_ppmi(p_i, p_j, p_ij):
    if p_ij == 0:
        return 0
    else:
        return max(math.log(p_ij / (p_i * p_j)), 0)


def calculate_npmi(p_i, p_j, p_ij):
    if p_ij == 0:
        return -1
    else:
        return math.log(p_ij / (p_i * p_j)) / math.log(p_ij)


def evaluate_performance(vocab_info_df: pd.DataFrame, result_df: pd.DataFrame, measure: str, top_k: int,
                         experiment: str):
    if measure != 'ppmi' and measure != 'npmi':
        raise Exception(f"No measure named {measure}")

    vocab_info_dict = vocab_info_df.set_index('vocab')['paper_ids'].to_dict()
    result_dict = result_df.set_index('seeds')['words'].to_dict()
    total_doc_count = len(set(element for entry in list(vocab_info_df.loc[:, 'paper_ids']) for element in entry))
    result_scores = np.zeros(len(result_dict))

    for idx, (seed, words) in enumerate(tqdm(result_dict.items(), desc=f"Evaluating Performance for {experiment}")):
        
        seed_scores = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, words=words, top_k=top_k, total_doc_count=total_doc_count)        
          
        # Aggregation
        result_scores[idx] = np.mean(seed_scores)
        
    print(result_dict.keys())
    return np.mean(result_scores)


def calculate_score_per_seed(vocab_info_dict: pd.DataFrame, words: list, top_k: int, total_doc_count: int):

    # Define variables for the function
    words = words[:top_k]
    segmented_words = list(permutations(words, 2))
    seed_scores = np.zeros(len(segmented_words))

    for idy, (w_i, w_j) in enumerate(segmented_words):    
        # Calculate criteria 
        p_i = len(vocab_info_dict[w_i]) / total_doc_count
        p_j = len(vocab_info_dict[w_j]) / total_doc_count
        c_ij = len(vocab_info_dict[w_i] & vocab_info_dict[w_j])
        p_ij = c_ij / total_doc_count

        if measure == 'ppmi':
            seed_scores[idy] = calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'npmi':
            seed_scores[idy] = calculate_npmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        else:
            raise Exception(f"No measure named {measure}")

    return seed_scores


if __name__ == '__main__':
    vocab_info_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_vocab.pkl"

    bert_base_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_bert_base_cosine_shortest_distance_ranked.pkl"
    scibert_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_scibert_cosine_shortest_distance_ranked.pkl"
    cosine_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_combined_shortest_distance_ranked.pkl"

    cosine_combined_pmi_npmi_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_combined_pmi_npmi_ranked.pkl"

    seetopic_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/seetopic_merged_result.pkl"

    measure = 'npmi'
    top_k = 10
    print(evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                               result_df=pd.read_pickle(bert_base_cosine_result_path), measure=measure,
                               top_k=top_k,
                               experiment='Cosine Bert Base'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                               result_df=pd.read_pickle(scibert_cosine_result_path), measure=measure,
                               top_k=top_k,
                               experiment='Cosine Scibert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                               result_df=pd.read_pickle(cosine_combined_result_path), measure=measure,
                               top_k=top_k,
                               experiment='Cosine Combined'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                               result_df=pd.read_pickle(cosine_combined_pmi_npmi_result_save_path),
                               top_k=top_k,
                               measure=measure,
                               experiment='Cosine Combined NPMI'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                               result_df=pd.read_pickle(seetopic_result_path), measure=measure, top_k=top_k,
                               experiment='Seetopic'))
