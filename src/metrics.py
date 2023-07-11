import math

import pandas as pd
from itertools import combinations

from tqdm import tqdm


def evaluate_performance(vocab_info_df: pd.DataFrame, result_df: pd.DataFrame, experiment: str):
    # if metric != 'pmi' and metric != 'npmi':
    #    raise Exception(f"no metric named {metric}")
    vocab_info_dict = vocab_info_df.set_index('vocab')['paper_ids'].to_dict()
    result_dict = result_df.set_index('seeds')['words'].to_dict()
    total_word_count = sum(len(value) for value in vocab_info_dict.values())

    # Starts intra seed pmi and npmi calculations
    pmi_sum = 0
    npmi_sum = 0
    for seed, words in tqdm(result_dict.items(), desc="Evaluating Performance"):
        for w_j, w_k in tqdm(combos := combinations(words, 2), total=len(combos),
                             desc=f"Evaluating Performance for Seed {seed}"):
            w_j_count = len(vocab_info_dict[w_j])
            w_k_count = len(vocab_info_dict[w_k])
            w_j_w_k_count = len(set(vocab_info_dict[w_j]) & set(vocab_info_dict[w_k]))
            if w_j_w_k_count == 0:
                pmi = -1
                npmi = -1
            else:
                pmi = math.log2((w_j_w_k_count * total_word_count) / (w_j_count * w_k_count))
                npmi = pmi / (math.log2(w_j_w_k_count / total_word_count) * -1)
            # Add to sum
            pmi_sum += pmi
            npmi_sum += npmi
    print(f"The pmi for {experiment} is {str(pmi_sum)} .")
    print(f"The npmi for {experiment} is {str(npmi_sum)} .")


if __name__ == "__main__":
    vocab_info_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_vocab_no_punc_no_special_char_keep_apos_hyphens.pkl"

    bert_base_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_shortest_distance.pkl"
    scibert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_shortest_distance.pkl"
    bert_base_scibert_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_combined_shortest_distance.pkl"
    min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_min_max_normalized_combined_shortest_distance.pkl"
    z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_z_score_normalized_combined_shortest_distance.pkl"

    bert_base_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_cosine_shortest_distance.pkl"
    scibert_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_cosine_shortest_distance.pkl"
    cosine_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_combined_shortest_distance.pkl"
    cosine_min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_min_max_normalized_combined_shortest_distance.pkl"
    cosine_z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_z_score_normalized_combined_shortest_distance.pkl"

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(bert_base_shortest_result_path),
                         experiment="Bert Base Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(bert_base_cosine_result_path),
                         experiment="Bert Base Cosine Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(scibert_shortest_result_path),
                         experiment="Scibert Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(scibert_cosine_result_path),
                         experiment="Scibert Cosine Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(bert_base_scibert_combined_result_path),
                         experiment="Bert Base Scibert Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_combined_result_path),
                         experiment="Bert Base Scibert Cosine Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(min_max_normalized_combined_result_path),
                         experiment="Bert Base Scibert Min Max Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_min_max_normalized_combined_result_path),
                         experiment="Bert Base Scibert Cosine Min Max Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(z_score_normalized_combined_result_path),
                         experiment="Bert Base Scibert Z Score Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_z_score_normalized_combined_result_path),
                         experiment="Bert Base Scibert Cosine Z Score Normalized Combined Shortest Distance")
