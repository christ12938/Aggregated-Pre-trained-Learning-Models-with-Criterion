import pandas as pd

pd.options.mode.use_inf_as_na = True
pd.set_option('display.width', 2000)


def agg_sets(series):
    merged = set()
    for s in series:
        merged.update(s)
    return merged


def merge_dicts(scidocs_path: str, amazon_path: str, twitter_path: str, legal_path: str, save_path: str):
    scidocs_df = pd.read_pickle(scidocs_path)
    amazon_df = pd.read_pickle(amazon_path)
    #twitter_df = pd.read_pickle(twitter_path)
    #legal_df = pd.read_pickle(legal_path)
    merged_df = pd.concat([scidocs_df, amazon_df]).groupby('vocab')['paper_ids'].agg(
        agg_sets).reset_index()
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    print(scidocs_df)
    print(amazon_df)
    #print(twitter_df)
    #print(legal_df)
    print(merged_df)
    merged_df.to_pickle(save_path)


if __name__ == "__main__":
    scidocs_vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/scidocs_vocab.pkl"
    amazon_vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/amazon_vocab.pkl"
    twitter_vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/twitter_vocab.pkl"
    legal_vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/legal_vocab.pkl"
    merged_vocab_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_vocab.pkl"
    merge_dicts(scidocs_path=scidocs_vocab_path, amazon_path=amazon_vocab_path, twitter_path=twitter_vocab_path,
                legal_path=legal_vocab_path,
                save_path=merged_vocab_save_path)
