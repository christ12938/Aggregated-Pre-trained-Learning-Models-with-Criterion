import pandas as pd
from tqdm import tqdm

from src.seeds import get_scidocs_seeds


def remove_vocabs_in_seed(seeds: list, vocabs_df: pd.DataFrame, save_path: str):
    vocabs_dict = vocabs_df.to_dict('records')
    indices_to_remove = []
    for idx, row in tqdm(enumerate(vocabs_dict), desc="Removing Vocabs"):
        for seed in seeds:
            if any(word in row['vocab'] for word in seed.split()):
                indices_to_remove.append(idx)
                break
    print(indices_to_remove)
    print(vocabs_df)
    vocabs_df = vocabs_df.drop(indices_to_remove).reset_index(drop=True)
    print(vocabs_df)
    vocabs_df.to_pickle(save_path)

if __name__ == "__main__":
    scidocs_vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/seetopic_scidocs_vocab.pkl"
    scidocs_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/seetopic_scidocs_vocab_trim.pkl"

    remove_vocabs_in_seed(seeds=get_scidocs_seeds(), vocabs_df=pd.read_pickle(scidocs_vocab_path), save_path=scidocs_save_path)