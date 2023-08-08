import pandas as pd
import chardet


class SeeTopicDataPreprocess:
    def __init__(self):
        self.idx = 0

    def create_vocab_list(self, data_path: str, save_path: str, prefix: str):
        self.idx = 0
        vocab_dict = {}
        with open(data_path, 'r') as file:
            for line in file:
                vocab_in_sentence = line.strip().split()
                for word in vocab_in_sentence:
                    vocab_dict.setdefault(word, set()).add(f'{prefix}{self.idx}')
                self.idx += 1
        vocab_df = pd.DataFrame(list(vocab_dict.items()), columns=['vocab', 'paper_ids'])
        vocab_df.to_pickle(save_path)
        print(vocab_df)


if __name__ == "__main__":
    scidocs_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/scidocs.txt"
    amazon_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/amazon.txt"
    twitter_data_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/twitter.txt"

    scidocs_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/scidocs_vocab.pkl"
    amazon_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/amazon_vocab.pkl"
    twitter_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/twitter_vocab.pkl"

    preproc = SeeTopicDataPreprocess()
    preproc.create_vocab_list(data_path=scidocs_data_path, save_path=scidocs_save_path, prefix="SCIDOCS")
    preproc.create_vocab_list(data_path=amazon_data_path, save_path=amazon_save_path, prefix="AMAZON")
    preproc.create_vocab_list(data_path=twitter_data_path, save_path=twitter_save_path, prefix="TWITTER")
