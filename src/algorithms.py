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
                distance = torch.Tensor([torch.abs(embeddings - seed_tensor).sum() for seed_tensor in seeds_tensors])
                score = torch.min(distance)
                result_index = torch.argmin(distance)
            else:
                distance = torch.Tensor([cosine_similarity(seed_tensor, embeddings) for seed_tensor in seeds_tensors])
                score = torch.max(distance)
                result_index = torch.argmax(distance)
            self.result_dict[seeds_words[result_index]].append((vocab, score))
        reverse = False if self.criteria == "manhattan" else True
        for seed, words in tqdm(self.result_dict.items(), desc="Ranking Vocabs for Shortest Distance"):
            sorted_vocabs = sorted(words, key=lambda x: x[1], reverse=reverse)
            self.result_dict[seed] = [vocab[0] for vocab in sorted_vocabs]


class CombinedShortestDistance(BaseAlgorithm):
    def __init__(self, vocabs_tensors_path_1: str, seed_embeddings_path_1: str, vocabs_tensors_path_2: str,
                 seed_embeddings_path_2: str, normalization=None, criteria="manhattan"):
        super().__init__(normalization=normalization)
        self.vocabs_df_1 = pd.read_pickle(vocabs_tensors_path_1)
        self.vocabs_df_2 = pd.read_pickle(vocabs_tensors_path_2)
        # self.vocabs_df_3 = pd.read_pickle(vocabs_tensors_path_3)
        self.seed_embeddings_df_1 = pd.read_pickle(seed_embeddings_path_1)
        self.seed_embeddings_df_2 = pd.read_pickle(seed_embeddings_path_2)
        # self.seed_embeddings_df_3 = pd.read_pickle(seed_embeddings_path_3)
        self.criteria = criteria

    def process_vocabs_embeddings(self):
        if self.normalization is not None:
            scalar_1 = self.normalize_fit(data=np.vstack(pd.concat(
                [self.seed_embeddings_df_1.loc[:, "embeddings"], self.vocabs_df_1.loc[:, "embeddings"]])))
            scalar_2 = self.normalize_fit(data=np.vstack(pd.concat(
                [self.seed_embeddings_df_2.loc[:, "embeddings"], self.vocabs_df_2.loc[:, "embeddings"]])))
            # scalar_3 = self.normalize_fit(data=np.vstack(pd.concat(
            #    [self.seed_embeddings_df_3.loc[:, "embeddings"], self.vocabs_df_3.loc[:, "embeddings"]])))
            tqdm.pandas(desc=f"Normalizing Vocabs Embeddings with {self.normalization} Normalization")
            self.vocabs_df_1['embeddings'] = self.vocabs_df_1['embeddings'].progress_apply(scalar_1.transform)
            self.vocabs_df_2['embeddings'] = self.vocabs_df_2['embeddings'].progress_apply(scalar_2.transform)
            # self.vocabs_df_3['embeddings'] = self.vocabs_df_3['embeddings'].progress_apply(scalar_3.transform)
            tqdm.pandas(desc=f"Normalizing Seeds Embeddings with {self.normalization} Normalization")
            self.seed_embeddings_df_1['embeddings'] = self.seed_embeddings_df_1['embeddings'].progress_apply(
                scalar_1.transform)
            self.seed_embeddings_df_2['embeddings'] = self.seed_embeddings_df_2['embeddings'].progress_apply(
                scalar_2.transform)
            # self.seed_embeddings_df_3['embeddings'] = self.seed_embeddings_df_3['embeddings'].progress_apply(
            #    scalar_3.transform)

        vocabs_dict_1 = self.vocabs_df_1.to_dict('records')
        vocabs_dict_2 = self.vocabs_df_2.to_dict('records')
        # vocabs_dict_3 = self.vocabs_df_3.to_dict('records')

        seeds_words = list(self.seed_embeddings_df_1.loc[:, "seeds"])

        seeds_tensors_1 = [torch.from_numpy(tensor).to(device) for tensor in
                           list(self.seed_embeddings_df_1.loc[:, "embeddings"])]
        seeds_tensors_2 = [torch.from_numpy(tensor).to(device) for tensor in
                           list(self.seed_embeddings_df_2.loc[:, "embeddings"])]
        # seeds_tensors_3 = [torch.from_numpy(tensor).to(device) for tensor in
        #                   list(self.seed_embeddings_df_3.loc[:, "embeddings"])]

        assert len(vocabs_dict_1) == len(vocabs_dict_2)  # == len(vocabs_dict_3)

        self.result_dict = {key: [] for key in seeds_words}
        for row_1, row_2 in tqdm(zip(vocabs_dict_1, vocabs_dict_2), total=len(vocabs_dict_1),
                                 desc="Processing Vocabs for Combined Shortest Distance"):
            vocab_1 = row_1['vocabs']
            vocab_2 = row_2['vocabs']
            # vocab_3 = row_3['vocabs']

            embeddings_1 = torch.from_numpy(row_1['embeddings']).to(device)
            embeddings_2 = torch.from_numpy(row_2['embeddings']).to(device)
            # embeddings_3 = torch.from_numpy(row_3['embeddings']).to(device)

            if self.criteria == "manhattan":
                distances_1 = torch.Tensor(
                    [torch.abs(embeddings_1 - seed_tensor).sum() for seed_tensor in seeds_tensors_1])
                distances_2 = torch.Tensor(
                    [torch.abs(embeddings_2 - seed_tensor).sum() for seed_tensor in seeds_tensors_2])
                # distances_3 = torch.Tensor(
                #    [torch.abs(embeddings_3 - seed_tensor).sum() for seed_tensor in seeds_tensors_3])

                min_1 = torch.min(distances_1)
                min_2 = torch.min(distances_2)
                # min_3 = torch.min(distances_3)

                # if min_1 <= min_2 and min_1 <= min_3:
                if min_1 <= min_2:
                    self.result_dict[seeds_words[torch.argmin(distances_1)]].append((vocab_1, min_1))
                # elif min_2 <= min_1 and min_2 <= min_3:
                else:
                    self.result_dict[seeds_words[torch.argmin(distances_2)]].append((vocab_2, min_2))
                # else:
                #    self.result_dict[seeds_words[torch.argmin(distances_3)]].append((vocab_3, min_3))

            else:
                distances_1 = torch.Tensor(
                    [cosine_similarity(seed_tensor, embeddings_1) for seed_tensor in seeds_tensors_1])
                distances_2 = torch.Tensor(
                    [cosine_similarity(seed_tensor, embeddings_2) for seed_tensor in seeds_tensors_2])
                # distances_3 = torch.Tensor(
                #    [cosine_similarity(seed_tensor, embeddings_3) for seed_tensor in seeds_tensors_3])

                max_1 = torch.max(distances_1)
                max_2 = torch.max(distances_2)
                # max_3 = torch.max(distances_3)

                # if max_1 >= max_2 and max_1 >= max_3:
                if max_1 >= max_2:
                    self.result_dict[seeds_words[torch.argmax(distances_1)]].append((vocab_1, max_1))
                else:
                    # elif max_2 >= max_1 and max_2 >= max_3:
                    self.result_dict[seeds_words[torch.argmax(distances_2)]].append((vocab_2, max_2))
                # else:
                #    self.result_dict[seeds_words[torch.argmax(distances_3)]].append((vocab_3, max_3))

        reverse = False if self.criteria == "manhattan" else True
        for seed, words in tqdm(self.result_dict.items(), desc="Ranking Vocabs for Combined Shortest Distance"):
            sorted_vocabs = sorted(words, key=lambda vocab: vocab[1].item(), reverse=reverse)
            self.result_dict[seed] = [vocab[0] for vocab in sorted_vocabs]


if __name__ == "__main__":  # "__cosined_normalized_combined_shortest_distance__":

    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_vocab_tensors_bert_base_uncased.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_vocab_tensors_scibert_uncased.pkl"
    #vocab_legalbert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_vocab_tensors_legalbert_uncased.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_bert_base_uncased.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_scibert_uncased.pkl"
    #seed_legalbert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/merged_seed_tensors_legalbert_uncased.pkl"

    bert_base_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_bert_base_shortest_distance_ranked.pkl"
    scibert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_scibert_shortest_distance_ranked.pkl"
    #legalbert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_legalbert_shortest_distance_ranked.pkl"
    combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_combined_shortest_distance_ranked.pkl"
    min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_min_max_normalized_combined_shortest_distance_ranked.pkl"
    #z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_z_score_normalized_combined_shortest_distance_ranked.pkl"

    bert_base_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_bert_base_cosine_shortest_distance_ranked.pkl"
    scibert_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_scibert_cosine_shortest_distance_ranked.pkl"
    #legalbert_cosine_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_legalbert_cosine_shortest_distance_ranked.pkl"
    cosine_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_combined_shortest_distance_ranked.pkl"
    #cosine_min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_min_max_normalized_combined_shortest_distance_ranked.pkl"
    #cosine_z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/new_result_data/merged_cosine_z_score_normalized_combined_shortest_distance_ranked.pkl"

    # __shortest_distance__

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path)
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_shortest_result_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path)
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_shortest_result_path)
    del scibert_result

    cosine_bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                               seed_embeddings_path=seed_bert_base_path, criteria="cosine")
    cosine_bert_base_result.process_vocabs_embeddings()
    cosine_bert_base_result.save_result(save_path=bert_base_cosine_result_path)
    del cosine_bert_base_result

    cosine_scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path,
                                             seed_embeddings_path=seed_scibert_path,
                                             criteria="cosine")
    cosine_scibert_result.process_vocabs_embeddings()
    cosine_scibert_result.save_result(save_path=scibert_cosine_result_path)
    del cosine_scibert_result

    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path)
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_path)
    del combined_result

    # __combined_shortest_distance__
    cosine_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                      vocabs_tensors_path_2=vocab_scibert_path,
                                                      seed_embeddings_path_1=seed_bert_base_path,
                                                      seed_embeddings_path_2=seed_scibert_path,
                                                      criteria="cosine")
    cosine_combined_result.process_vocabs_embeddings()
    cosine_combined_result.save_result(save_path=cosine_combined_result_path)
    del cosine_combined_result

    # __normalized_combined_shortest_distance__
    min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  normalization="min_max")

    min_max_normalized_combined_result.process_vocabs_embeddings()
    min_max_normalized_combined_result.save_result(save_path=min_max_normalized_combined_result_path)
    del min_max_normalized_combined_result

    """
    legalbert_result = ShortestDistance(vocabs_tensors_path=vocab_legalbert_path,
                                        seed_embeddings_path=seed_legalbert_path)
    legalbert_result.process_vocabs_embeddings()
    legalbert_result.save_result(save_path=legalbert_shortest_result_path)
    del legalbert_result

    # __combined_shortest_distance__
    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               vocabs_tensors_path_3=vocab_legalbert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path,
                                               seed_embeddings_path_3=seed_legalbert_path)
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_path)
    del combined_result
    # __normalized_combined_shortest_distance__
    min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  vocabs_tensors_path_3=vocab_legalbert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  seed_embeddings_path_3=seed_legalbert_path,
                                                                  normalization="min_max")

    min_max_normalized_combined_result.process_vocabs_embeddings()
    min_max_normalized_combined_result.save_result(save_path=min_max_normalized_combined_result_path)
    del min_max_normalized_combined_result

    z_score_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                  vocabs_tensors_path_2=vocab_scibert_path,
                                                                  vocabs_tensors_path_3=vocab_legalbert_path,
                                                                  seed_embeddings_path_1=seed_bert_base_path,
                                                                  seed_embeddings_path_2=seed_scibert_path,
                                                                  seed_embeddings_path_3=seed_legalbert_path,
                                                                  normalization="z_score")
    z_score_normalized_combined_result.process_vocabs_embeddings()
    z_score_normalized_combined_result.save_result(save_path=z_score_normalized_combined_result_path)
    del z_score_normalized_combined_result

    # __shortest_distance__

    cosine_bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                               seed_embeddings_path=seed_bert_base_path, criteria="cosine")
    cosine_bert_base_result.process_vocabs_embeddings()
    cosine_bert_base_result.save_result(save_path=bert_base_cosine_result_path)
    del cosine_bert_base_result

    cosine_scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path,
                                             seed_embeddings_path=seed_scibert_path,
                                             criteria="cosine")
    cosine_scibert_result.process_vocabs_embeddings()
    cosine_scibert_result.save_result(save_path=scibert_cosine_result_path)
    del cosine_scibert_result

    cosine_legalbert_result = ShortestDistance(vocabs_tensors_path=vocab_legalbert_path,
                                               seed_embeddings_path=seed_legalbert_path,
                                               criteria="cosine")
    cosine_legalbert_result.process_vocabs_embeddings()
    cosine_legalbert_result.save_result(save_path=legalbert_cosine_result_path)
    del cosine_legalbert_result

    # __combined_shortest_distance__
    cosine_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                      vocabs_tensors_path_2=vocab_scibert_path,
                                                      vocabs_tensors_path_3=vocab_legalbert_path,
                                                      seed_embeddings_path_1=seed_bert_base_path,
                                                      seed_embeddings_path_2=seed_scibert_path,
                                                      seed_embeddings_path_3=seed_legalbert_path, criteria="cosine")
    cosine_combined_result.process_vocabs_embeddings()
    cosine_combined_result.save_result(save_path=cosine_combined_result_path)
    del cosine_combined_result

    # __normalized_combined_shortest_distance__
    cosine_min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                         vocabs_tensors_path_2=vocab_scibert_path,
                                                                         vocabs_tensors_path_3=vocab_legalbert_path,
                                                                         seed_embeddings_path_1=seed_bert_base_path,
                                                                         seed_embeddings_path_2=seed_scibert_path,
                                                                         seed_embeddings_path_3=seed_legalbert_path,
                                                                         normalization="min_max", criteria="cosine")

    cosine_min_max_normalized_combined_result.process_vocabs_embeddings()
    cosine_min_max_normalized_combined_result.save_result(save_path=cosine_min_max_normalized_combined_result_path)
    del cosine_min_max_normalized_combined_result

    cosine_z_score_normalized_combined_result_ranked = CombinedShortestDistance(
        vocabs_tensors_path_1=vocab_bert_base_path,
        vocabs_tensors_path_2=vocab_scibert_path,
        vocabs_tensors_path_3=vocab_legalbert_path,
        seed_embeddings_path_1=seed_bert_base_path,
        seed_embeddings_path_2=seed_scibert_path,
        seed_embeddings_path_3=seed_legalbert_path,
        normalization="z_score", criteria="cosine")
    cosine_z_score_normalized_combined_result_ranked.process_vocabs_embeddings()
    cosine_z_score_normalized_combined_result_ranked.save_result(
        save_path=cosine_z_score_normalized_combined_result_path)
    del cosine_z_score_normalized_combined_result_ranked


if __name__ == "__main__":
    vocab_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_vocab_tensors_bert_base.pkl"
    vocab_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_vocab_tensors_scibert.pkl"

    seed_bert_base_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_bert_base.pkl"
    seed_scibert_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seed_tensors_scibert.pkl"

    bert_base_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_cosine_ranked.pkl"
    scibert_shortest_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_scibert_cosine_ranked.pkl"
    combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_combined_cosine_ranked.pkl"
    cosine_min_max_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_scibert_cosine_min_max_normalized_combined_shortest_distance_ranked.pkl"
    cosine_z_score_normalized_combined_result_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/seetopic_bert_base_scibert_cosine_z_score_normalized_combined_shortest_distance_ranked.pkl"

    bert_base_result = ShortestDistance(vocabs_tensors_path=vocab_bert_base_path,
                                        seed_embeddings_path=seed_bert_base_path, criteria="cosine")
    bert_base_result.process_vocabs_embeddings()
    bert_base_result.save_result(save_path=bert_base_shortest_result_path)
    del bert_base_result

    scibert_result = ShortestDistance(vocabs_tensors_path=vocab_scibert_path, seed_embeddings_path=seed_scibert_path,
                                      criteria="cosine")
    scibert_result.process_vocabs_embeddings()
    scibert_result.save_result(save_path=scibert_shortest_result_path)
    del scibert_result

    # __combined_shortest_distance__
    combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                               vocabs_tensors_path_2=vocab_scibert_path,
                                               seed_embeddings_path_1=seed_bert_base_path,
                                               seed_embeddings_path_2=seed_scibert_path, criteria="cosine")
    combined_result.process_vocabs_embeddings()
    combined_result.save_result(save_path=combined_result_path)
    del combined_result

    # __normalized_combined_shortest_distance__
    cosine_min_max_normalized_combined_result = CombinedShortestDistance(vocabs_tensors_path_1=vocab_bert_base_path,
                                                                         vocabs_tensors_path_2=vocab_scibert_path,
                                                                         seed_embeddings_path_1=seed_bert_base_path,
                                                                         seed_embeddings_path_2=seed_scibert_path,
                                                                         normalization="min_max", criteria="cosine")

    cosine_min_max_normalized_combined_result.process_vocabs_embeddings()
    cosine_min_max_normalized_combined_result.save_result(save_path=cosine_min_max_normalized_combined_result_path)
    del cosine_min_max_normalized_combined_result

    cosine_z_score_normalized_combined_result_ranked = CombinedShortestDistance(
        vocabs_tensors_path_1=vocab_bert_base_path,
        vocabs_tensors_path_2=vocab_scibert_path,
        seed_embeddings_path_1=seed_bert_base_path,
        seed_embeddings_path_2=seed_scibert_path,
        normalization="z_score", criteria="cosine")
    cosine_z_score_normalized_combined_result_ranked.process_vocabs_embeddings()
    cosine_z_score_normalized_combined_result_ranked.save_result(
        save_path=cosine_z_score_normalized_combined_result_path)
    del cosine_z_score_normalized_combined_result_ranked
"""
