import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.functional import cosine_similarity

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'


class BaseAlgorithm:
    def __init__(self, normalization=None):
        self.result_dict = None
        if normalization == "min_max" or normalization == "z_score" or normalization is None:
            self.normalization = normalization
        else:
            raise Exception(f"no normalization method called {normalization}")

    def normalize_fit(self, data):
        if self.normalization == "min_max":
            return MinMaxScaler().fit(data)
        else:
            return StandardScaler().fit(data)

    def save_result(self, save_path: str):
        print("Saving Results ...")
        pd.DataFrame(list(self.result_dict.items()), columns=['seeds', 'words']).to_pickle(save_path)


class ShortestDistance(BaseAlgorithm):
    def __init__(self, vocabs_tensors_path: str, seed_embeddings_path: str, normalization=None, criteria="manhattan"):
        super().__init__(normalization=normalization)
        self.vocabs_df = pd.read_pickle(vocabs_tensors_path)
        self.seed_embeddings_df = pd.read_pickle(seed_embeddings_path)
        self.criteria = criteria

    def process_vocabs_embeddings(self):
        if self.normalization is not None:
            scalar = self.normalize_fit(data=np.vstack(pd.concat(
                [self.seed_embeddings_df.loc[:, "embeddings"], self.vocabs_df.loc[:, "embeddings"]])))
            tqdm.pandas(desc=f"Normalizing Vocabs Embeddings with {self.normalization} Normalization")
            self.vocabs_df['embeddings'] = self.vocabs_df['embeddings'].apply(scalar.transform)
            tqdm.pandas(desc=f"Normalizing Seeds Embeddings with {self.normalization} Normalization")
            self.seed_embeddings_df['embeddings'] = self.seed_embeddings_df['embeddings'].apply(scalar.transform)

        vocabs_dict = self.vocabs_df.to_dict('records')
        seeds_words = list(self.seed_embeddings_df.loc[:, "seeds"])
        seeds_tensors = [torch.from_numpy(tensor).to(device) for tensor in
                         list(self.seed_embeddings_df.loc[:, "embeddings"])]
        self.result_dict = {key: [] for key in seeds_words}
        for row in tqdm(vocabs_dict, desc="Processing Vocabs for Shortest Distance"):
            vocab = row['vocabs']
            embeddings = torch.from_numpy(row['embeddings']).to(device)
            if self.criteria == "manhattan":
                result_index = torch.argmin(
                    torch.Tensor([torch.abs(embeddings - seed_tensor).sum() for seed_tensor in seeds_tensors]))
            else:
                result_index = torch.argmin(
                    torch.Tensor([cosine_similarity(seed_tensor, embeddings) for seed_tensor in seeds_tensors]))
            self.result_dict[seeds_words[result_index]].append(vocab)


class CombinedShortestDistance(BaseAlgorithm):
    def __init__(self, vocabs_tensors_path_1: str, seed_embeddings_path_1: str, vocabs_tensors_path_2: str,
                 seed_embeddings_path_2: str, normalization=None, criteria="manhattan"):
        super().__init__(normalization=normalization)
        self.vocabs_df_1 = pd.read_pickle(vocabs_tensors_path_1)
        self.vocabs_df_2 = pd.read_pickle(vocabs_tensors_path_2)
        self.seed_embeddings_df_1 = pd.read_pickle(seed_embeddings_path_1)
        self.seed_embeddings_df_2 = pd.read_pickle(seed_embeddings_path_2)
        self.criteria = criteria

    def process_vocabs_embeddings(self):
        if self.normalization is not None:
            scalar_1 = self.normalize_fit(data=np.vstack(pd.concat(
                [self.seed_embeddings_df_1.loc[:, "embeddings"], self.vocabs_df_1.loc[:, "embeddings"]])))
            scalar_2 = self.normalize_fit(data=np.vstack(pd.concat(
                [self.seed_embeddings_df_2.loc[:, "embeddings"], self.vocabs_df_2.loc[:, "embeddings"]])))
            tqdm.pandas(desc=f"Normalizing Vocabs Embeddings with {self.normalization} Normalization")
            self.vocabs_df_1['embeddings'] = self.vocabs_df_1['embeddings'].progress_apply(scalar_1.transform)
            self.vocabs_df_2['embeddings'] = self.vocabs_df_2['embeddings'].progress_apply(scalar_2.transform)
            tqdm.pandas(desc=f"Normalizing Seeds Embeddings with {self.normalization} Normalization")
            self.seed_embeddings_df_1['embeddings'] = self.seed_embeddings_df_1['embeddings'].progress_apply(
                scalar_1.transform)
            self.seed_embeddings_df_2['embeddings'] = self.seed_embeddings_df_2['embeddings'].progress_apply(
                scalar_2.transform)

        vocabs_dict_1 = self.vocabs_df_1.to_dict('records')
        vocabs_dict_2 = self.vocabs_df_2.to_dict('records')

        seeds_words = list(self.seed_embeddings_df_1.loc[:, "seeds"])

        seeds_tensors_1 = [torch.from_numpy(tensor).to(device) for tensor in
                           list(self.seed_embeddings_df_1.loc[:, "embeddings"])]
        seeds_tensors_2 = [torch.from_numpy(tensor).to(device) for tensor in
                           list(self.seed_embeddings_df_2.loc[:, "embeddings"])]

        assert len(vocabs_dict_1) == len(vocabs_dict_2)

        self.result_dict = {key: [] for key in seeds_words}

        for row_1, row_2 in tqdm(zip(vocabs_dict_1, vocabs_dict_2), total=len(vocabs_dict_1),
                                 desc="Processing Vocabs for Combined Shortest Distance"):
            vocab_1 = row_1['vocabs']
            vocab_2 = row_2['vocabs']

            embeddings_1 = torch.from_numpy(row_1['embeddings']).to(device)
            embeddings_2 = torch.from_numpy(row_2['embeddings']).to(device)

            if self.criteria == "manhattan":
                distances_1 = torch.Tensor(
                    [torch.abs(embeddings_1 - seed_tensor).sum() for seed_tensor in seeds_tensors_1])
                distances_2 = torch.Tensor(
                    [torch.abs(embeddings_2 - seed_tensor).sum() for seed_tensor in seeds_tensors_2])
            else:
                distances_1 = torch.Tensor(
                    [cosine_similarity(seed_tensor, embeddings_1) for seed_tensor in seeds_tensors_1])
                distances_2 = torch.Tensor(
                    [cosine_similarity(seed_tensor, embeddings_2) for seed_tensor in seeds_tensors_2])

            if torch.min(distances_1) <= torch.min(distances_2):
                self.result_dict[seeds_words[torch.argmin(distances_1)]].append(vocab_1)
            else:
                self.result_dict[seeds_words[torch.argmin(distances_2)]].append(vocab_2)


"""
if __name__ == "__shortest_distance__":
    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_shortest_distance.pkl"
    scibert_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_shortest_distance.pkl"

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path)
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_result_save_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path)
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_result_save_path)
    del scibert_result

if __name__ == "__combined_shortest_distance__":
    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_combined_shortest_distance.pkl"

    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path)
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_save_path)
    del combined_result

if __name__ == "__main__":  # "__normalized_combined_shortest_distance__":

    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_shortest_distance.pkl"
    scibert_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_shortest_distance.pkl"

    combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_combined_shortest_distance.pkl"

    min_max_normalized_combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_min_max_normalized_combined_shortest_distance.pkl"
    z_score_normalized_combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_z_score_normalized_combined_shortest_distance.pkl"

    # __shortest_distance__

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path)
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_result_save_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path)
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_result_save_path)
    del scibert_result

    # __combined_shortest_distance__
    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path)
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_save_path)
    del combined_result

    # __normalized_combined_shortest_distance__
    min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  normalization="min_max")

    min_max_normalized_combined_result.process_vocabs_embeddings()
    min_max_normalized_combined_result.save_result(save_path=min_max_normalized_combined_result_save_path)
    del min_max_normalized_combined_result

    z_score_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  normalization="z_score")
    z_score_normalized_combined_result.process_vocabs_embeddings()
    z_score_normalized_combined_result.save_result(save_path=z_score_normalized_combined_result_save_path)
    del z_score_normalized_combined_result
"""

if __name__ == "__main__":  # "__cosined_normalized_combined_shortest_distance__":

    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_cosine_shortest_distance.pkl"
    scibert_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/scibert_cosine_shortest_distance.pkl"

    combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_combined_shortest_distance.pkl"

    min_max_normalized_combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_min_max_normalized_combined_shortest_distance.pkl"
    z_score_normalized_combined_result_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/bert_base_scibert_cosine_z_score_normalized_combined_shortest_distance.pkl"

    # __shortest_distance__

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path, criteria="cosine")
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_result_save_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path,
                                      criteria="cosine")
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_result_save_path)
    del scibert_result

    # __combined_shortest_distance__
    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path, criteria="cosine")
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_save_path)
    del combined_result

    # __normalized_combined_shortest_distance__
    min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  normalization="min_max", criteria="cosine")

    min_max_normalized_combined_result.process_vocabs_embeddings()
    min_max_normalized_combined_result.save_result(save_path=min_max_normalized_combined_result_save_path)
    del min_max_normalized_combined_result

    z_score_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  normalization="z_score", criteria="cosine")
    z_score_normalized_combined_result.process_vocabs_embeddings()
    z_score_normalized_combined_result.save_result(save_path=z_score_normalized_combined_result_save_path)
    del z_score_normalized_combined_result
