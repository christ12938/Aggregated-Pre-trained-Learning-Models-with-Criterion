import pandas as pd

from src.seeds import get_amazon_seeds, get_scidocs_seeds, get_merged_list_seeds


def seetopic_postproc(result_path: str, seed_list: list, save_path: str):
    words_list = []
    with open(result_path, 'r') as file:
        for line in file:
            if ':' in line:
                after_colon = line.split(':', 1)[-1].strip()
                words = after_colon.split(',')
                words_list.append(words[1:])
    pd.DataFrame({'seeds': seed_list, 'words': words_list}).to_pickle(save_path)


if __name__ == '__main__':
    seetopic_amazon_result_path = "/home/chris/SeeTopic/amazon/keywords_seetopic.txt"
    seetopic_amazon_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/seetopic_amazon_result.pkl"
    seetopic_scidocs_result_path = "/home/chris/SeeTopic/scidocs/keywords_seetopic.txt"
    seetopic_scidocs_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/seetopic_scidocs_result.pkl"
    seetopic_merged_result_path = "/home/chris/SeeTopic/merged/keywords_seetopic.txt"
    seetopic_merged_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/seetopic_merged_result.pkl"
    #seetopic_postproc(result_path=seetopic_amazon_result_path, seed_list=get_amazon_seeds(),
    #                  save_path=seetopic_amazon_result_save_path)

    #seetopic_postproc(result_path=seetopic_scidocs_result_path, seed_list=get_scidocs_seeds(),
    #                  save_path=seetopic_scidocs_result_save_path)

    seetopic_postproc(result_path=seetopic_merged_result_path, seed_list=get_merged_list_seeds(),
                      save_path=seetopic_merged_result_save_path)
