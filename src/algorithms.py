import pandas as pd
import torch
from tqdm import tqdm

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'


class ShortestDistance:
    def __init__(self, vocabs_tensors_path: str, seed_embeddings_path: str):
        self.vocabs_df = pd.read_pickle(vocabs_tensors_path)
        self.seed_embeddings_df = pd.read_pickle(seed_embeddings_path)
        self.result_dict = None

    def process_vocabs_embeddings(self):
        vocabs_dict = self.vocabs_df.to_dict('records')
        seeds_words = list(self.seed_embeddings_df.loc[:, "seeds"])
        seeds_tensors = [torch.from_numpy(tensor[0]).to(device) for tensor in
                         list(self.seed_embeddings_df.loc[:, "embeddings"])]
        self.result_dict = {key: [] for key in seeds_words}
        for row in tqdm(vocabs_dict):
            vocab = row['vocabs']
            embeddings = torch.from_numpy(row['embeddings'][0]).to(device)
            result_index = torch.argmin(
                torch.Tensor([torch.abs(embeddings - seed_tensor).sum() for seed_tensor in seeds_tensors]))
            self.result_dict[seeds_words[result_index]].append(vocab)

    def save_result(self, save_path: str):
        print("Saving Results ...")
        pd.DataFrame(list(self.result_dict.items()), columns=['seeds', 'words']).to_csv(save_path)


if __name__ == "__main__":
    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_shortest_distance.csv"
    scibert_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_shortest_distance.csv"

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path)
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_result_save_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path)
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_result_save_path)
    del scibert_result
