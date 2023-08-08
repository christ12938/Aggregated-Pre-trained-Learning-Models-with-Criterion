import math
import sys

import pandas as pd
from itertools import combinations

from tqdm import tqdm


def evaluate_performance(vocab_info_df: pd.DataFrame, result_df: pd.DataFrame, experiment: str):
    # if metric != 'pmi' and metric != 'npmi':
    #    raise Exception(f"no metric named {metric}")
    vocab_info_dict = vocab_info_df.set_index('vocab')['paper_ids'].to_dict()
    result_dict = result_df.set_index('seeds')['words'].to_dict()
    total_doc_count = len(set(element for entry in list(vocab_info_df.loc[:, 'paper_ids']) for element in entry))

    # Starts intra seed pmi and npmi calculations
    ppmi_sum = npmi_sum = 0
    for seed, words in tqdm(result_dict.items(), desc=f"Evaluating Performance for {experiment}"):
        words = words[:10]
        ppmi = npmi = 0
        for w_i, w_j in list(combinations(words, 2)):
            p_i = len(vocab_info_dict[w_i]) / total_doc_count
            p_j = len(vocab_info_dict[w_j]) / total_doc_count
            c_ij = len(vocab_info_dict[w_i] & vocab_info_dict[w_j])
            p_ij = c_ij / total_doc_count

            if p_ij == 0:
                ppmi += 0
                npmi -= 1
            else:
                # PMI
                # pmi += math.log(p_ij / (p_i * p_j))

                # PPMI
                ppmi += max(math.log(p_ij / (p_i * p_j)), 0)

                # NPMI
                npmi -= math.log(p_ij / (p_i * p_j)) / math.log(p_ij)

                # LCP
                # lcp += math.log(p_ij / p_j)

        # pmi_sum += (pmi / math.comb(len(words), 2))
        ppmi_sum += (ppmi / math.comb(len(words), 2))
        npmi_sum += (npmi / math.comb(len(words), 2))
        #print(f"{seed}: PMI = {(ppmi / math.comb(len(words), 2))} NPMI = {(npmi / math.comb(len(words), 2))}")
        # lcp_sum += (lcp / math.comb(len(words), 2))

    # pmi_sum = pmi_sum / len(result_dict.keys())
    ppmi_sum = ppmi_sum / len(result_dict.keys())
    npmi_sum = npmi_sum / len(result_dict.keys())
    # lcp_sum = lcp_sum / len(result_dict.keys())

    sys.stderr.flush()
    print(experiment)
    # print('PMI:', pmi_sum)
    print('PPMI:', ppmi_sum)
    print('NPMI:', npmi_sum)
    # print('LCP:', lcp_sum)


def evaluate_combined_result(vocab_info_df: pd.DataFrame, result_df_1: pd.DataFrame, result_df_2: pd.DataFrame,
                             experiment: str, save_path: str):
    vocab_info_dict = vocab_info_df.set_index('vocab')['paper_ids'].to_dict()
    total_doc_count = len(set(element for entry in list(vocab_info_df.loc[:, 'paper_ids']) for element in entry))
    result_dict_1 = result_df_1.set_index('seeds')['words'].to_dict()
    result_dict_2 = result_df_2.set_index('seeds')['words'].to_dict()
    #result_dict_3 = result_df_3.set_index('seeds')['words'].to_dict()

    # Starts intra seed pmi and npmi calculations
    result_dict = {}
    ppmi_sum = npmi_sum = 0
    for (seed_1, words_1), (seed_2, words_2) in tqdm(
            zip(sorted(result_dict_1.items()), sorted(result_dict_2.items())),
            total=len(result_df_1), desc=f"Evaluating Performance for {experiment}"):
        #words_list = [words_1[:10], words_2[:10], words_3[:10]]
        words_list = [words_1[:10], words_2[:10]]
        assert (seed_1 == seed_2)
        max_ppmi = max_npmi = float('-inf')
        max_npmi_result = None
        for words in words_list:
            ppmi = npmi = 0
            for w_i, w_j in list(combinations(words, 2)):
                p_i = len(vocab_info_dict[w_i]) / total_doc_count
                p_j = len(vocab_info_dict[w_j]) / total_doc_count
                c_ij = len(vocab_info_dict[w_i] & vocab_info_dict[w_j])
                p_ij = c_ij / total_doc_count

                if p_ij == 0:
                    ppmi += 0
                    npmi -= 1
                else:
                    # PMI
                    # pmi += math.log(p_ij / (p_i * p_j))

                    # PPMI
                    ppmi += max(math.log(p_ij / (p_i * p_j)), 0)

                    # NPMI
                    npmi -= math.log(p_ij / (p_i * p_j)) / math.log(p_ij)

                # LCP
                # lcp += math.log(p_ij / p_j)

            # pmi_sum += (pmi / math.comb(len(words), 2))
            ppmi = (ppmi / math.comb(len(words), 2))
            npmi = (npmi / math.comb(len(words), 2))

            if ppmi > max_ppmi:
                max_ppmi = ppmi

            if npmi > max_npmi:
                max_npmi = npmi
                max_npmi_result = words

        ppmi_sum += max_ppmi
        npmi_sum += max_npmi
        result_dict[seed_1] = max_npmi_result

    ppmi_sum = ppmi_sum / len(result_dict.keys())
    npmi_sum = npmi_sum / len(result_dict.keys())

    sys.stderr.flush()
    print(experiment)
    # print('PMI:', pmi_sum)
    print('PPMI:', ppmi_sum)
    print('NPMI:', npmi_sum)
    # print('LCP:', lcp_sum)
    pd.DataFrame(list(result_dict.items()), columns=['seed', 'words']).to_pickle(save_path)


"""
if __name__ == "__main__":
    test_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/temp/merged_test.txt"
    topk = 20

    bert_base_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_cosine_ranked.pkl"
    scibert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_scibert_cosine_ranked.pkl"
    combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_combined_cosine_ranked.pkl"
    cosine_min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_scibert_cosine_min_max_normalized_combined_shortest_distance_ranked.pkl"
    cosine_z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_scibert_cosine_z_score_normalized_combined_shortest_distance_ranked.pkl"

    seetopic_metric(result_path=bert_base_shortest_result_path, test_data_path=test_data_path, topk=topk,
                    experiment="Seetopic Bert Base Cosine Shortest Distance")

    seetopic_metric(result_path=scibert_shortest_result_path, test_data_path=test_data_path, topk=topk,
                    experiment="Seetopic Scibert Cosine Shortest Distance")

    seetopic_metric(result_path=combined_result_path, test_data_path=test_data_path, topk=topk,
                    experiment="Seetopic Bert Base Scibert Cosine Combined Shortest Distance")

    seetopic_metric(result_path=cosine_min_max_normalized_combined_result_path, test_data_path=test_data_path,
                    topk=topk,
                    experiment="Seetopic Bert Base Scibert Cosine Min Max Normalized Combined Shortest Distance")

    seetopic_metric(result_path=cosine_z_score_normalized_combined_result_path, test_data_path=test_data_path,
                    topk=topk,
                    experiment="Seetopic Bert Base Scibert Cosine Z Score Normalized Combined Shortest Distance")

"""
if __name__ == "__main__":
    vocab_info_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_vocab.pkl"

    bert_base_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_bert_base_shortest_distance_ranked.pkl"
    scibert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_scibert_shortest_distance_ranked.pkl"
    min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_min_max_normalized_combined_shortest_distance_ranked.pkl"

    bert_base_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_bert_base_cosine_shortest_distance_ranked.pkl"
    scibert_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_scibert_cosine_shortest_distance_ranked.pkl"
    cosine_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_combined_shortest_distance_ranked.pkl"

    combined_pmi_npmi_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_combined_pmi_npmi_ranked.pkl"
    cosine_combined_pmi_npmi_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_combined_pmi_npmi_ranked.pkl"

    seetopic_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/seetopic_merged_result.pkl"

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
                         result_df=pd.read_pickle(min_max_normalized_combined_result_path),
                         experiment="Min Max Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_combined_result_path),
                         experiment="Cosine Combined Shortest Distance")

    evaluate_combined_result(vocab_info_df=pd.read_pickle(vocab_info_path),
                             result_df_1=pd.read_pickle(bert_base_shortest_result_path),
                             result_df_2=pd.read_pickle(scibert_shortest_result_path),
                             experiment="Combined PMI NPMI",
                             save_path=combined_pmi_npmi_result_save_path)

    evaluate_combined_result(vocab_info_df=pd.read_pickle(vocab_info_path),
                             result_df_1=pd.read_pickle(bert_base_cosine_result_path),
                             result_df_2=pd.read_pickle(scibert_cosine_result_path),
                             experiment="Cosine combined PMI NPMI",
                             save_path=cosine_combined_pmi_npmi_result_save_path)

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(seetopic_result_path),
                         experiment="Seetopic")
    """
    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(legalbert_shortest_result_path),
                         experiment="Legalbert Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(legalbert_cosine_result_path),
                         experiment="Legalbert Cosine Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(combined_result_path),
                         experiment="Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_combined_result_path),
                         experiment="Cosine Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(min_max_normalized_combined_result_path),
                         experiment="Min Max Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_min_max_normalized_combined_result_path),
                         experiment="Cosine Min Max Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(z_score_normalized_combined_result_path),
                         experiment="Z Score Normalized Combined Shortest Distance")

    evaluate_performance(vocab_info_df=pd.read_pickle(vocab_info_path),
                         result_df=pd.read_pickle(cosine_z_score_normalized_combined_result_path),
                         experiment="Cosine Z Score Normalized Combined Shortest Distance")

    evaluate_combined_result(vocab_info_df=pd.read_pickle(vocab_info_path),
                             result_df_1=pd.read_pickle(bert_base_shortest_result_path),
                             result_df_2=pd.read_pickle(scibert_shortest_result_path),
                             result_df_3=pd.read_pickle(legalbert_shortest_result_path),
                             experiment="Combined PMI NPMI basic",
                             save_path=combined_pmi_npmi_result_save_path)

    evaluate_combined_result(vocab_info_df=pd.read_pickle(vocab_info_path),
                             result_df_1=pd.read_pickle(bert_base_cosine_result_path),
                             result_df_2=pd.read_pickle(scibert_cosine_result_path),
                             result_df_3=pd.read_pickle(legalbert_cosine_result_path),
                             experiment="Cosine combined PMI NPMI",
                             save_path=cosine_combined_pmi_npmi_result_save_path)
    """
