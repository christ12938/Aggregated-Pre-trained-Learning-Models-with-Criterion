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

    vocab_info_dict = vocab_info_df.set_index('vocab')['id'].to_dict()
    result_dict = result_df.set_index('seed')['words'].to_dict()
    total_doc_count = len(set(element for entry in list(vocab_info_df.loc[:, 'id']) for element in entry))
    result_scores = np.zeros(len(result_dict))

    for idx, (seed, words) in enumerate(tqdm(result_dict.items(), desc=f"Evaluating Performance for {experiment}")):
        
        result_scores[idx] = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, words=words, 
                                                      top_k=top_k, total_doc_count=total_doc_count, measure=measure)        
          
    return np.mean(result_scores)


def calculate_score_per_seed(vocab_info_dict: pd.DataFrame, words: list, top_k: int, total_doc_count: int, measure: str):

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

    # Aggregation
    return np.mean(seed_scores)


if __name__ == '__main__':
    
    # Vocab info paths
    scidocs_vocab_path = "data/scidocs_vocab.pkl"
    amazon_vocab_path = "data/amazon_vocab.pkl"
    french_vocab_path = "data/xlsum_fr_vocab.pkl"
    merged_vocab_path = "data/merged_vocab.pkl"

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
    
    measure = 'npmi'
    top_k = 10

    # Scidocs
    print(evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                               result_df=pd.read_pickle(scidocs_bert_base_result_save_path), measure=measure,
                               top_k=top_k,
                               experiment='Scidocs Cosine Bert Base'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                               result_df=pd.read_pickle(scidocs_scibert_result_save_path), measure=measure,
                               top_k=top_k,
                               experiment='Scidocs Cosine Scibert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                               result_df=pd.read_pickle(scidocs_flaubert_result_save_path), measure=measure,
                               top_k=top_k,
                               experiment='Scidocs Cosine Flaubert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                               result_df=pd.read_pickle(scidocs_combined_npmi_result_save_path), measure=measure,
                               top_k=top_k,
                               experiment='Scidocs Combined NPMI'))
    
    # Amazon
    print(evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                               result_df=pd.read_pickle(amazon_bert_base_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Amazon Cosine Bert Base'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                               result_df=pd.read_pickle(amazon_scibert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Amazon Cosine Scibert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                               result_df=pd.read_pickle(amazon_flaubert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Amazon Cosine Flaubert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                               result_df=pd.read_pickle(amazon_combined_npmi_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Amazon Combined NPMI'))

    # XLSum FR
    print(evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                               result_df=pd.read_pickle(french_bert_base_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='XLSum FR Cosine Bert Base'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                               result_df=pd.read_pickle(french_scibert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='XLSum FR Cosine Scibert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                               result_df=pd.read_pickle(french_flaubert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='XLSum FR Cosine Flaubert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                               result_df=pd.read_pickle(french_combined_npmi_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='XLSum FR Combined NPMI'))
    
    # Merged
    print(evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                               result_df=pd.read_pickle(merged_bert_base_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Merged Cosine Bert Base'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                               result_df=pd.read_pickle(merged_scibert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Merged Cosine Scibert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                               result_df=pd.read_pickle(merged_flaubert_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Merged Cosine Flaubert'))
    print(evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                               result_df=pd.read_pickle(merged_combined_npmi_result_save_path), measure=measure, 
                               top_k=top_k,
                               experiment='Merged Combined NPMI'))
