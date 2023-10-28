import math

import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import permutations, combinations
from sklearn.metrics.pairwise import cosine_similarity
from utils import CRITERIA_LIST


def calculate_ppmi(p_i, p_j, p_ij):
    if p_ij == 0:
        return 0
    else:
        return max(math.log(p_ij / (p_i * p_j)), 0)


def calculate_npmi(p_i, p_j, p_ij):
    if p_ij == 0:
        return -1
    else:
        return -1 * (math.log(p_ij / (p_i * p_j)) / math.log(p_ij))


def calculate_delta(c_i, c_j, c_ij):
    min_freq = min(c_i, c_j)
    return (c_ij / (c_ij + 1)) * (min_freq / (min_freq + 1))


def calculate_wappmi(measure: str, vocab_info_dict: dict, doc_info_dict: dict, w_i: str, w_j: str, 
                     total_vocab_count: int, total_doc_count: int, total_word_count: int, intersecting_keys, p_i, p_j, p_ij):
    ppmi = calculate_ppmi(p_i, p_j, p_ij)
    if 'wappmi_alpha_1' in measure:
        return p_ij * ppmi 
    else:
        p_w_i_d_i = 0
        for intersect_key in intersecting_keys:
            if 'laplace' in measure:
                w_i_d_i_count = vocab_info_dict[w_i]['id'][intersect_key] + 0.01
            else:
                w_i_d_i_count = vocab_info_dict[w_i]['id'][intersect_key]
            p_w_i_d_i += (w_i_d_i_count / doc_info_dict[intersect_key]);

        if 'alpha_2' in measure:
            alpha_2 = 1 / (total_word_count - vocab_info_dict[w_i]['count'])
            return alpha_2 * p_w_i_d_i * ppmi
        elif 'alpha_3' in measure:
            alpha_3 = 1 / (vocab_info_dict[w_j]['count'] * total_vocab_count)
            return alpha_3 * p_w_i_d_i * ppmi
        else:
            return None


# TODO: Change
def evaluate_performance(vocab_info_df: pd.DataFrame, result_df: pd.DataFrame, measure: str, top_k: int,
                         experiment: str):
    if measure not in CRITERIA_LIST:
        raise Exception(f"No measure named {measure}")

    vocab_info_dict = vocab_info_df.set_index('vocab')['id'].to_dict()
    result_dict = result_df.set_index('seed')['words'].to_dict()
    total_doc_count = len(set(element for entry in list(vocab_info_df.loc[:, 'id']) for element in entry))
    result_scores = np.zeros(len(result_dict))

    for idx, (seed, words) in enumerate(tqdm(result_dict.items(), desc=f"Evaluating Performance for {experiment}")):
            
        result_scores[idx] = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, words=words, 
                                                      top_k=top_k, total_doc_count=total_doc_count, measure=measure)        
    return np.mean(result_scores)


def calculate_score_per_seed(vocab_info_dict: dict, doc_info_dict: dict, words: list, top_k: int, total_vocab_count: int, 
                             total_doc_count: int, total_word_count: int, measure: str):

    # Define variables for the function
    words = words[:top_k]
    segmented_words = list(combinations(words, 2))
    seed_scores = np.zeros(len(segmented_words))

    for idy, (w_i, w_j) in enumerate(segmented_words):    
        
        intersecting_keys = set(vocab_info_dict[w_i]['id'].keys()) & set(vocab_info_dict[w_j]['id'].keys())
        
        # Calculate criteria 
        if 'laplace' in measure:
            p_i = (len(vocab_info_dict[w_i]['id']) + 0.01) / total_doc_count
            p_j = (len(vocab_info_dict[w_j]['id']) + 0.01) / total_doc_count
            c_ij = len(intersecting_keys) + 0.01
            p_ij = c_ij / total_doc_count
        else:
            p_i = len(vocab_info_dict[w_i]['id']) / total_doc_count
            p_j = len(vocab_info_dict[w_j]['id']) / total_doc_count
            c_ij = len(intersecting_keys)
            p_ij = c_ij / total_doc_count
            
        if 'delta' in measure:
            delta = calculate_delta(len(vocab_info_dict[w_i]['id']), len(vocab_info_dict[w_j]['id']), c_ij)

        if measure == 'pmi_laplace' or measure == 'ppmi' or measure == 'ppmi_laplace':
            seed_scores[idy] = calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'ppmi_delta':
            seed_scores[idy] = delta * calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'npmi' or measure == 'npmi_laplace':
            seed_scores[idy] = calculate_npmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif 'wappmi' in measure:
            seed_scores[idy] = calculate_wappmi(measure=measure, vocab_info_dict=vocab_info_dict, 
                                                doc_info_dict=doc_info_dict,
                                                w_i=w_i, w_j=w_j, total_vocab_count=total_vocab_count, 
                                                total_doc_count=total_doc_count,
                                                total_word_count=total_word_count,
                                                intersecting_keys=intersecting_keys,
                                                p_i=p_i, p_j=p_j, p_ij=p_ij) 
            if 'delta' in measure:
                seed_scores[idy] = delta * seed_scores[idy]
        else:
            raise Exception(f"No measure named {measure}")

    # Aggregation
    return np.mean(seed_scores)


if __name__ == '__main__':
    
    # Vocab info paths
    scidocs_vocab_path = "data/scidocs_vocab.pkl"
    amazon_vocab_path = "data/amazon_vocab.pkl"
    french_vocab_path = "data/french_news_vocab.pkl"
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
    
    measure = 'wappmi_alpha_3'
    top_k = 20

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
