import math

import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import permutations, combinations
from sklearn.metrics.pairwise import cosine_similarity
from utils import CRITERIA_LIST

def calculate_info(vocab_info_dict: dict, words: list):
    result_doc = set()
    result_word = 0
    for word in words:
        result_doc.update(vocab_info_dict[word]['id'].keys())
        result_word += vocab_info_dict[word]['count']
    return len(result_doc), result_word

def calculate_pmi(p_i, p_j, p_ij):
    return math.log(p_ij / (p_i * p_j))


def calculate_pmi_smoothing(p_i, p_j, p_ij):
    # k = 3
    return math.log((p_ij * p_ij * p_ij) / (p_i * p_j))


def calculate_ppmi(p_i, p_j, p_ij):
    if p_ij == 0:
        return 0
    else:
        return max(math.log(p_ij / (p_i * p_j)), 0)


def calculate_npmi(p_i, p_j, p_ij):
    if p_ij == 0 or p_ij == 1:
        return -1
    else:
        return -1 * (math.log(p_ij / (p_i * p_j)) / math.log(p_ij))


def calculate_delta(c_i, c_j, c_ij):
    min_freq = min(c_i, c_j)
    return (c_ij / (c_ij + 1)) * (min_freq / (min_freq + 1))


def calculate_wa(measure: str, vocab_info_dict: dict, doc_info_dict: dict, w_i: str, w_j: str, 
                     total_vocab_count: int, total_doc_count: int, total_word_count: int, intersecting_keys, p_ij):
    if 'alpha_1' in measure:
        return p_ij 
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
            return alpha_2 * p_w_i_d_i
        elif 'alpha_3' in measure:
            alpha_3 = 1 / (vocab_info_dict[w_j]['count'] * total_vocab_count)
            return alpha_3 * p_w_i_d_i
        else:
            return None


def evaluate_performance(vocab_info_df: pd.DataFrame, scores_info_df: pd.DataFrame, doc_info_df: pd.DataFrame, 
                         result_df: pd.DataFrame, measure: str, top_k: int,
                         dataset: str):
    if measure not in CRITERIA_LIST:
        raise Exception(f"No measure named {measure}")

    vocab_info_dict = vocab_info_df.set_index('vocab').T.to_dict()
    scores_info_dict = scores_info_df.set_index('vocab').T.to_dict()
    doc_info_dict = doc_info_df.set_index('id')['length'].to_dict()
    result_dict = result_df.set_index('seed')['words'].to_dict()

    total_vocab_count = len(vocab_info_dict)
    total_doc_count = len(doc_info_dict)
    total_word_count = vocab_info_df['count'].sum()

    result_scores = []
    result_subset_scores = []

    for seed, words in tqdm(result_dict.items(), desc=f"Evaluating Performance for {dataset} {measure}"):
        
        total_vocab_count_subset = len(words[:top_k])
        total_doc_count_subset, total_word_count_subset = calculate_info(vocab_info_dict=vocab_info_dict, words=words[:top_k])

        inner_result_scores = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, doc_info_dict=doc_info_dict,
                                                            words=words[:top_k], total_vocab_count=total_vocab_count,
                                                            total_doc_count=total_doc_count, total_word_count=total_word_count,
                                                            measure=measure)
        inner_result_subset_scores = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, doc_info_dict=doc_info_dict,
                                                            words=words[:top_k], total_vocab_count=total_vocab_count_subset,
                                                            total_doc_count=total_doc_count_subset, total_word_count=total_word_count_subset,
                                                            measure=measure)
        result_scores.append(sum(y for x, y in inner_result_scores) / len(inner_result_scores))
        result_subset_scores.append(sum(y for x, y in inner_result_subset_scores) / len(inner_result_subset_scores))

    result_score = sum(result_scores) / len(result_scores) 
    result_subset_score =  sum(result_subset_scores) / len(result_subset_scores)
    print(f"Mean {measure} for {dataset} is {result_score:.3g}")
    print(f"Mean Subset {measure} for {dataset} is {result_subset_score:.3g}")
    print()

def calculate_score_per_seed(vocab_info_dict: dict, doc_info_dict: dict, words: list, total_vocab_count: int, 
                             total_doc_count: int, total_word_count: int, measure: str):

    # Define variables for the function
    segmented_words = list(combinations(words, 2))
    seed_scores = []

    for idy, (w_i, w_j) in enumerate(segmented_words):    
        seed_score = 0 
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

        if measure == 'pmi_laplace':
            seed_score = calculate_pmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'pmi_smoothing_laplace':
            seed_score = calculate_pmi_smoothing(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'ppmi' or measure == 'ppmi_laplace':
            seed_score = calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'ppmi_delta':
            seed_score = delta * calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif measure == 'npmi' or measure == 'npmi_laplace':
            seed_score = calculate_npmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
        elif 'wapmi' in measure or 'wappmi' in measure:
            seed_score = calculate_wa(measure=measure, vocab_info_dict=vocab_info_dict, 
                                            doc_info_dict=doc_info_dict,
                                            w_i=w_i, w_j=w_j, total_vocab_count=total_vocab_count, 
                                            total_doc_count=total_doc_count,
                                            total_word_count=total_word_count,
                                            intersecting_keys=intersecting_keys,
                                            p_ij=p_ij) 
            if 'wapmi' in measure:
                if 'smoothing' in measure:
                    seed_score = seed_score * calculate_pmi_smoothing(p_i=p_i, p_j=p_j, p_ij=p_ij)
                else:
                    seed_score = seed_score * calculate_pmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
            elif 'wappmi' in measure:
                seed_score = seed_score * calculate_ppmi(p_i=p_i, p_j=p_j, p_ij=p_ij)
                if 'delta' in measure:
                    seed_score = delta * seed_score
        else:
            raise Exception(f"No measure named {measure}")
        seed_scores.append((w_i, seed_score))

    # Aggregation
    return seed_scores


if __name__ == '__main__':
    
    top_k = 20

    # Vocab info paths
    scidocs_vocab_path = "data/scidocs_vocab.pkl"
    amazon_vocab_path = "data/amazon_vocab.pkl"
    french_vocab_path = "data/french_news_vocab.pkl"
    merged_vocab_path = "data/merged_vocab.pkl"

    # Scores info paths
    scidocs_scores_path = "data/scidocs_scores.pkl"
    amazon_scores_path = "data/amazon_scores.pkl"
    french_scores_path = "data/french_news_scores.pkl"
    merged_scores_path = "data/merged_scores.pkl"

    # Doc info paths
    scidocs_doc_path = "data/scidocs_doc.pkl"
    amazon_doc_path = "data/amazon_doc.pkl"
    french_doc_path = "data/french_news_doc.pkl"
    merged_doc_path = "data/merged_doc.pkl"


    for outer_criteria in CRITERIA_LIST:

        # Result paths
        scidocs_bert_base_result_save_path = "result_data/scidocs_result_bert_base_uncased.pkl"
        scidocs_scibert_result_save_path = "result_data/scidocs_result_scibert_uncased.pkl"
        scidocs_flaubert_result_save_path = "result_data/scidocs_result_flaubert_uncased.pkl"
        scidocs_seetopic_result_save_path = f"result_data/scidocs_result_seetopic.pkl"

        amazon_bert_base_result_save_path = "result_data/amazon_result_bert_base_uncased.pkl"
        amazon_scibert_result_save_path = "result_data/amazon_result_scibert_uncased.pkl"
        amazon_flaubert_result_save_path = "result_data/amazon_result_flaubert_uncased.pkl"
        amazon_seetopic_result_save_path = f"result_data/amazon_result_seetopic.pkl"

        french_bert_base_result_save_path = "result_data/french_result_bert_base_uncased.pkl"
        french_scibert_result_save_path = "result_data/french_result_scibert_uncased.pkl"
        french_flaubert_result_save_path = "result_data/french_result_flaubert_uncased.pkl"
        french_seetopic_result_save_path = f"result_data/french_result_seetopic.pkl"

        merged_bert_base_result_save_path = "result_data/merged_result_bert_base_uncased.pkl"
        merged_scibert_result_save_path = "result_data/merged_result_scibert_uncased.pkl"
        merged_flaubert_result_save_path = "result_data/merged_result_flaubert_uncased.pkl"
        merged_seetopic_result_save_path = f"result_data/merged_result_seetopic.pkl"
       
        
        # Scidocs
        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_bert_base_result_save_path), measure=outer_criteria,
                                   top_k=top_k,
                                   dataset='Scidocs bert Base')
        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_scibert_result_save_path), measure=outer_criteria,
                                   top_k=top_k,
                                   dataset='Scidocs Scibert')
        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_flaubert_result_save_path), measure=outer_criteria,
                                   top_k=top_k,
                                   dataset='Scidocs Flaubert')

        # Amazon

        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_bert_base_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Amazon Bert Base')
        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_scibert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Amazon Scibert')
        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_flaubert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Amazon Flaubert')

        # French News
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_bert_base_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='French News Bert Base')
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_scibert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='French News Scibert')
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_flaubert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='French News Flaubert')
        # Merged

        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_bert_base_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Merged Bert Base')
        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_scibert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Merged Scibert')
        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_flaubert_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Merged Flaubert')

        # Seetopic

        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_seetopic_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Scidocs Seetopic')
        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_seetopic_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Amazon Seetopic')
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_seetopic_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='French News Seetopic')
        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_seetopic_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset='Merged Seetopic')

        #for inner_criteria in CRITERIA_LIST:
        inner_criteria = 'npmi'

        scidocs_combined_npmi_result_save_path = f"result_data/scidocs_result_combined_{inner_criteria}.pkl"
        amazon_combined_npmi_result_save_path = f"result_data/amazon_result_combined_{inner_criteria}.pkl"
        french_combined_npmi_result_save_path = f"result_data/french_result_combined_{inner_criteria}.pkl"
        merged_combined_npmi_result_save_path = f"result_data/merged_result_combined_{inner_criteria}.pkl"

        scidocs_combined_scores_embeds_npmi_result_save_path = f"result_data/scidocs_result_combined_scores_embeds_{inner_criteria}.pkl"
        amazon_combined_scores_embeds_npmi_result_save_path = f"result_data/amazon_result_combined_scores_embeds_{inner_criteria}.pkl"
        french_combined_scores_embeds_npmi_result_save_path = f"result_data/french_result_combined_scores_embeds_{inner_criteria}.pkl"
        merged_combined_scores_embeds_npmi_result_save_path = f"result_data/merged_result_combined_scores_embeds_{inner_criteria}.pkl"

        # Scidocs
        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_combined_npmi_result_save_path), measure=outer_criteria,
                                   top_k=top_k,
                                   dataset=f'Scidocs Combined {inner_criteria}')
        evaluate_performance(vocab_info_df=pd.read_pickle(scidocs_vocab_path),
                            scores_info_df=pd.read_pickle(scidocs_scores_path),
                            doc_info_df=pd.read_pickle(scidocs_doc_path),
                                   result_df=pd.read_pickle(scidocs_combined_scores_embeds_npmi_result_save_path), measure=outer_criteria,
                                   top_k=top_k,
                                   dataset=f'Scidocs Combined Scores Embeds {inner_criteria}')
        
        # Amazon
        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_combined_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'Amazon Combined {inner_criteria}')
        evaluate_performance(vocab_info_df=pd.read_pickle(amazon_vocab_path),
                            scores_info_df=pd.read_pickle(amazon_scores_path),
                            doc_info_df=pd.read_pickle(amazon_doc_path),
                                   result_df=pd.read_pickle(amazon_combined_scores_embeds_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'Amazon Combined Scores Embeds {inner_criteria}')

        # French News
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_combined_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'French News Combined {inner_criteria}')
        evaluate_performance(vocab_info_df=pd.read_pickle(french_vocab_path),
                            scores_info_df=pd.read_pickle(french_scores_path),
                            doc_info_df=pd.read_pickle(french_doc_path),
                                   result_df=pd.read_pickle(french_combined_scores_embeds_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'French News Combined Scores Embeds {inner_criteria}')

        # Merged
        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_combined_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'Merged Combined {inner_criteria}')
        evaluate_performance(vocab_info_df=pd.read_pickle(merged_vocab_path),
                            scores_info_df=pd.read_pickle(merged_scores_path),
                            doc_info_df=pd.read_pickle(merged_doc_path),
                                   result_df=pd.read_pickle(merged_combined_scores_embeds_npmi_result_save_path), measure=outer_criteria, 
                                   top_k=top_k,
                                   dataset=f'Merged Combined Scores Embeds {inner_criteria}')
