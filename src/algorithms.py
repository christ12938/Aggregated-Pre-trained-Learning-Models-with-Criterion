import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import create_folder, CRITERIA_LIST
from metrics import calculate_score_per_seed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.functional import cosine_similarity


device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'


class CosineScore:
    def __init__(self, vocab_embeddings_path: str, seed_embeddings_path: str):
        self.vocab_embeddings_df = pd.read_pickle(vocab_embeddings_path)
        self.seed_embeddings_df = pd.read_pickle(seed_embeddings_path)
        self.result_dict = None

    def process_vocab_embeddings(self):
        vocab_embeddings_dict = self.vocab_embeddings_df.to_dict('records')
        seeds_words = list(self.seed_embeddings_df.loc[:, "seed"])
        seeds_embeddings_list = [torch.from_numpy(tensor).to(device) for tensor in
                                 list(self.seed_embeddings_df.loc[:, "embedding"])]
        self.result_dict = {key: [] for key in seeds_words}
       
        for row in tqdm(vocab_embeddings_dict, desc="Calculating Cosine Scores for Vocabs"):
            vocab = row['vocab']
            vocab_embedding = torch.from_numpy(row['embedding']).to(device)
           
            # calculate cosine scores
            cosine_scores = torch.Tensor([cosine_similarity(seed_embedding, vocab_embedding) for seed_embedding in seeds_embeddings_list])
            max_cosine_score = torch.max(cosine_scores)
            result_index = torch.argmax(cosine_scores)
            # append data to result
            self.result_dict[seeds_words[result_index]].append((vocab, max_cosine_score))

        # Sort scores based on cosine score
        for seed, words in tqdm(self.result_dict.items(), desc="Ranking Vocabs According to Cosine Scores"):
            sorted_vocabs = sorted(words, key=lambda x: x[1], reverse=True)
            self.result_dict[seed] = [vocab[0] for vocab in sorted_vocabs]


    def save_result(self, save_path: str):
        print("Saving Results ...")
        create_folder(path=save_path)
        pd.DataFrame(list(self.result_dict.items()), columns=['seed', 'words']).to_pickle(save_path)


class CombineCriteria:
    def __init__(self, vocab_info_path: str, doc_info_path: str, categorized_words_paths: list, model_list: list, criteria: str, top_k: int):
        if criteria not in CRITERIA_LIST:
            raise Exception(f"No criteria named {criteria}")
        self.criteria = criteria
        self.top_k = top_k
        self.vocab_info_df = None
        self.doc_info_df = None
        self.categorized_words_dicts = None
        self.model_list = model_list
        self.result_dict = None
        self.decision_dict = None
        self.open_files(vocab_info_path=vocab_info_path, doc_info_path=doc_info_path, categorized_words_paths=categorized_words_paths)
    

    def open_files(self, vocab_info_path: str, doc_info_path: str, categorized_words_paths: list):
        self.vocab_info_df = pd.read_pickle(vocab_info_path)
        self.doc_info_df = pd.read_pickle(doc_info_path)
        self.categorized_words_dicts = [pd.read_pickle(path).set_index('seed')['words'].to_dict() for path in categorized_words_paths] 
        
        # Sanity Checks: TODO
    

    def combine_results(self):
        vocab_info_dict = self.vocab_info_df.set_index('vocab').T.to_dict()
        doc_info_dict = self.doc_info_df.set_index('id')['length'].to_dict() 
#        total_vocab_count = len(vocab_info_dict)
#        total_doc_count = len(self.doc_info_df['id'])
#        total_word_count = self.vocab_info_df['count'].sum() 
        seeds = list(self.categorized_words_dicts[0].keys())
        self.result_dict = {}
        self.decision_dict = {}

        for seed in tqdm(seeds, desc=f"Calculating Highest {self.top_k} {self.criteria} for Each Seed"):
            seed_scores = np.zeros(len(self.categorized_words_dicts))
            for idx, result in enumerate(self.categorized_words_dicts):
                total_doc_count, total_word_count = calculate_info(vocab_info_dict, result[seed][:top_k]) 
                temp = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, doc_info_dict=doc_info_dict, 
                        words=result[seed][:top_k], total_vocab_count=top_k, 
                                                            total_doc_count=total_doc_count, total_word_count=total_word_count, 
                                                            measure=self.criteria)
                seed_scores[idx] = sum(t[1] for t in temp) / len(temp)
            self.result_dict[seed] = self.categorized_words_dicts[np.argmax(seed_scores)][seed]
            self.decision_dict[seed] = self.model_list[np.argmax(seed_scores)]


    def save_result(self, result_save_path: str, decision_save_path: str):
        print("Saving Results ...")
        create_folder(path=result_save_path)
        create_folder(path=decision_save_path)
        pd.DataFrame(list(self.result_dict.items()), columns=['seed', 'words']).to_pickle(result_save_path)
        pd.DataFrame(list(self.decision_dict.items()), columns=['seed', 'decision']).to_pickle(decision_save_path)


def calculate_info(vocab_info_dict: dict, words: list):
    result_doc = set()
    result_word = 0
    for word in words:
        result_doc.update(vocab_info_dict[word]['id'].keys())
        result_word += vocab_info_dict[word]['count']
    return len(result_doc), result_word


class CombineScoresEmbeddings:
    def __init__(self, vocab_info_path: str, scores_info_path: str, doc_info_path: str, categorized_words_path: str, criteria: str, top_k: int, discard_subtopic=False):
        if criteria not in CRITERIA_LIST:
            raise Exception(f"No criteria named {criteria}")
        self.criteria = criteria
        self.top_k = top_k
        self.scores_info_df = None
        self.vocab_info_df = None
        self.doc_info_df = None
        self.categorized_words_dict = None
        self.result_dict = None
        self.discard_subtopic = discard_subtopic
        self.open_files(vocab_info_path=vocab_info_path, scores_info_path=scores_info_path, doc_info_path=doc_info_path, categorized_words_path=categorized_words_path)
    

    def open_files(self, vocab_info_path: str, scores_info_path: str, doc_info_path: str, categorized_words_path: str):
        self.vocab_info_df = pd.read_pickle(vocab_info_path)
        self.scores_info_df = pd.read_pickle(scores_info_path)
        self.doc_info_df = pd.read_pickle(doc_info_path)
        self.categorized_words_dict = pd.read_pickle(categorized_words_path).set_index('seed')['words'].to_dict() 
        
        # Sanity Checks: TODO


    @staticmethod
    def remove_duplicates_based_on_first(input_list):
        seen = set()
        result = []
        for item in input_list:
            # Check if the first element of the tuple is in the set
            if item[0] not in seen:
                seen.add(item[0])
                result.append(item)
        return result

    @staticmethod
    def sum_based_on_first(tuples_list):
        result = {}
        for key, value in tuples_list:
            result.setdefault(key, 0)
            result[key] += value
        return list(result.items())

    def combine_results(self):
        vocab_info_dict = self.vocab_info_df.set_index('vocab').T.to_dict()
        doc_info_dict = self.doc_info_df.set_index('id')['length'].to_dict() 
        scores_info_dict = self.scores_info_df.set_index('vocab').T.to_dict()
        self.result_dict = {}

        for seed, words in tqdm(self.categorized_words_dict.items(), desc=f"Processing Highest {self.top_k} {self.criteria} for Each Seed"):
            result_words = []
            top_k_words = words[:top_k]            
            for word in top_k_words:
                subtopic_words = []
                #topic_score = scores_info_dict[word][f'total_{self.criteria}_score']
                if self.discard_subtopic is False:
                    result_words.append(word)
                result_words.extend([key for key, val in scores_info_dict[word][f'{self.criteria}_candidate'].items()])
#                subtopic_words.extend([key for key, val in scores_info_dict[word][f'{self.criteria}_candidate'].items()])
#                total_doc_count, total_word_count = calculate_info(vocab_info_dict, subtopic_words) 
#                temp = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, doc_info_dict=doc_info_dict, 
#                                                            words=subtopic_words, total_vocab_count=len(subtopic_words), 
#                                                            total_doc_count=total_doc_count, total_word_count=total_word_count, 
#                                                            measure=self.criteria)
#                temp_score = sum(t[1] for t in temp) / len(temp)
#                result_words.append((subtopic_words, temp_score * scores_info_dict[word][f'total_{self.criteria}_score']))
            #TODO: How to sort?
            #result_words = list(set(result_words)) 
            #total_doc_count, total_word_count = calculate_info(vocab_info_dict, result_words) 
            #result_words = self.remove_duplicates_based_on_first(result_words)
#            result_words = calculate_score_per_seed(vocab_info_dict=vocab_info_dict, doc_info_dict=doc_info_dict, 
#                                                        words=result_words, total_vocab_count=len(result_words), 
#                                                        total_doc_count=total_doc_count, total_word_count=total_word_count, 
#                                                        measure=self.criteria)
#            result_words = self.sum_based_on_first(result_words)
#            result_words = sorted(result_words, key=lambda x: x[1], reverse=True)
#            result_words = [e[0] if isinstance(e, tuple) else e for e in result_words]
#            self.result_dict[seed] = [e[0] if isinstance(e, tuple) else e for e in result_words]
            seed_scores = [scores_info_dict[e][f'total_{self.criteria}_score'] for e in result_words]
            result_words = [x for _, x in sorted(zip(seed_scores, result_words), reverse=True)]
            self.result_dict[seed] = []
            for item in result_words:
                if item not in self.result_dict[seed]:
                    self.result_dict[seed].append(item)


    def save_result(self, result_save_path: str):
        print("Saving Results ...")
        create_folder(path=result_save_path)
        pd.DataFrame(list(self.result_dict.items()), columns=['seed', 'words']).to_pickle(result_save_path)


def perform_algorithms():

    # Seeds Embeddings
    scidocs_seed_embed_bert_base_save_path = "embeddings/scidocs_seed_tensors_bert_large_uncased.pkl"
    scidocs_seed_embed_scibert_save_path = "embeddings/scidocs_seed_tensors_scibert_uncased.pkl"
    scidocs_seed_embed_flaubert_save_path = "embeddings/scidocs_seed_tensors_flaubert_large_uncased.pkl"

    amazon_seed_embed_bert_base_save_path = "embeddings/amazon_seed_tensors_bert_large_uncased.pkl"
    amazon_seed_embed_scibert_save_path = "embeddings/amazon_seed_tensors_scibert_uncased.pkl"
    amazon_seed_embed_flaubert_save_path = "embeddings/amazon_seed_tensors_flaubert_large_uncased.pkl"

    french_seed_embed_bert_base_save_path = "embeddings/french_seed_tensors_bert_large_uncased.pkl"
    french_seed_embed_scibert_save_path = "embeddings/french_seed_tensors_scibert_uncased.pkl"
    french_seed_embed_flaubert_save_path = "embeddings/french_seed_tensors_flaubert_large_uncased.pkl"

    merged_seed_embed_bert_base_save_path = "embeddings/merged_seed_tensors_bert_large_uncased.pkl"
    merged_seed_embed_scibert_save_path = "embeddings/merged_seed_tensors_scibert_uncased.pkl"
    merged_seed_embed_flaubert_save_path = "embeddings/merged_seed_tensors_flaubert_large_uncased.pkl"

    
    # Vocab Embeddings
    scidocs_vocab_embed_bert_base_save_path = "embeddings/scidocs_vocab_tensors_bert_large_uncased.pkl"
    scidocs_vocab_embed_scibert_save_path = "embeddings/scidocs_vocab_tensors_scibert_uncased.pkl"
    scidocs_vocab_embed_flaubert_save_path = "embeddings/scidocs_vocab_tensors_flaubert_large_uncased.pkl"

    amazon_vocab_embed_bert_base_save_path = "embeddings/amazon_vocab_tensors_bert_large_uncased.pkl"
    amazon_vocab_embed_scibert_save_path = "embeddings/amazon_vocab_tensors_scibert_uncased.pkl"
    amazon_vocab_embed_flaubert_save_path = "embeddings/amazon_vocab_tensors_flaubert_large_uncased.pkl"

    french_vocab_embed_bert_base_save_path = "embeddings/french_vocab_tensors_bert_large_uncased.pkl"
    french_vocab_embed_scibert_save_path = "embeddings/french_vocab_tensors_scibert_uncased.pkl"
    french_vocab_embed_flaubert_save_path = "embeddings/french_vocab_tensors_flaubert_large_uncased.pkl"

    merged_vocab_embed_bert_base_save_path = "embeddings/merged_vocab_tensors_bert_large_uncased.pkl"
    merged_vocab_embed_scibert_save_path = "embeddings/merged_vocab_tensors_scibert_uncased.pkl"
    merged_vocab_embed_flaubert_save_path = "embeddings/merged_vocab_tensors_flaubert_large_uncased.pkl"

   
    # Result paths
    scidocs_bert_base_result_save_path = "result_data/scidocs_result_bert_base_uncased.pkl"
    scidocs_scibert_result_save_path = "result_data/scidocs_result_scibert_uncased.pkl"
    scidocs_flaubert_result_save_path = "result_data/scidocs_result_flaubert_uncased.pkl"

    amazon_bert_base_result_save_path = "result_data/amazon_result_bert_base_uncased.pkl"
    amazon_scibert_result_save_path = "result_data/amazon_result_scibert_uncased.pkl"
    amazon_flaubert_result_save_path = "result_data/amazon_result_flaubert_uncased.pkl"

    french_bert_base_result_save_path = "result_data/french_result_bert_base_uncased.pkl"
    french_scibert_result_save_path = "result_data/french_result_scibert_uncased.pkl"
    french_flaubert_result_save_path = "result_data/french_result_flaubert_uncased.pkl"

    merged_bert_base_result_save_path = "result_data/merged_result_bert_base_uncased.pkl"
    merged_scibert_result_save_path = "result_data/merged_result_scibert_uncased.pkl"
    merged_flaubert_result_save_path = "result_data/merged_result_flaubert_uncased.pkl"


    # Scidocs
    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_bert_base_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_bert_base_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_bert_base_result_save_path)
    del scidocs_max_cosine
 
    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_scibert_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_scibert_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_scibert_result_save_path)
    del scidocs_max_cosine
    
    scidocs_max_cosine = CosineScore(vocab_embeddings_path=scidocs_vocab_embed_flaubert_save_path, 
                                     seed_embeddings_path=scidocs_seed_embed_flaubert_save_path)
    scidocs_max_cosine.process_vocab_embeddings()
    scidocs_max_cosine.save_result(save_path=scidocs_flaubert_result_save_path)
    del scidocs_max_cosine
    
    # Amazon
    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_bert_base_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_bert_base_result_save_path)
    del amazon_max_cosine
 
    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_scibert_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_scibert_result_save_path)
    del amazon_max_cosine
 
    amazon_max_cosine = CosineScore(vocab_embeddings_path=amazon_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=amazon_seed_embed_flaubert_save_path)
    amazon_max_cosine.process_vocab_embeddings()
    amazon_max_cosine.save_result(save_path=amazon_flaubert_result_save_path)
    del amazon_max_cosine
    
    # XLSum FR
    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=french_seed_embed_bert_base_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_bert_base_result_save_path)
    del french_max_cosine
    
    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=french_seed_embed_scibert_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_scibert_result_save_path)
    del french_max_cosine
 
    french_max_cosine = CosineScore(vocab_embeddings_path=french_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=french_seed_embed_flaubert_save_path)
    french_max_cosine.process_vocab_embeddings()
    french_max_cosine.save_result(save_path=french_flaubert_result_save_path)
    del french_max_cosine
 
    # Merged
    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_bert_base_save_path, 
                                    seed_embeddings_path=merged_seed_embed_bert_base_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_bert_base_result_save_path)
    del merged_max_cosine
 
    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_scibert_save_path, 
                                    seed_embeddings_path=merged_seed_embed_scibert_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_scibert_result_save_path)
    del merged_max_cosine
 
    merged_max_cosine = CosineScore(vocab_embeddings_path=merged_vocab_embed_flaubert_save_path, 
                                    seed_embeddings_path=merged_seed_embed_flaubert_save_path)
    merged_max_cosine.process_vocab_embeddings()
    merged_max_cosine.save_result(save_path=merged_flaubert_result_save_path)
    del merged_max_cosine


if __name__ == "__main__":  


    #perform_algorithms()

    for criteria in CRITERIA_LIST:

        top_k = 10

        # Result paths
        scidocs_bert_base_result_save_path = "result_data/scidocs_result_bert_base_uncased.pkl"
        scidocs_scibert_result_save_path = "result_data/scidocs_result_scibert_uncased.pkl"
        scidocs_flaubert_result_save_path = "result_data/scidocs_result_flaubert_uncased.pkl"
        scidocs_combined_result_save_path = f"result_data/scidocs_result_combined_{criteria}.pkl"

        amazon_bert_base_result_save_path = "result_data/amazon_result_bert_base_uncased.pkl"
        amazon_scibert_result_save_path = "result_data/amazon_result_scibert_uncased.pkl"
        amazon_flaubert_result_save_path = "result_data/amazon_result_flaubert_uncased.pkl"
        amazon_combined_result_save_path = f"result_data/amazon_result_combined_{criteria}.pkl"

        french_bert_base_result_save_path = "result_data/french_result_bert_base_uncased.pkl"
        french_scibert_result_save_path = "result_data/french_result_scibert_uncased.pkl"
        french_flaubert_result_save_path = "result_data/french_result_flaubert_uncased.pkl"
        french_combined_result_save_path = f"result_data/french_result_combined_{criteria}.pkl"

        merged_bert_base_result_save_path = "result_data/merged_result_bert_base_uncased.pkl"
        merged_scibert_result_save_path = "result_data/merged_result_scibert_uncased.pkl"
        merged_flaubert_result_save_path = "result_data/merged_result_flaubert_uncased.pkl"
        merged_combined_result_save_path = f"result_data/merged_result_combined_{criteria}.pkl"

     
        # Combined NPMI method
        scidocs_vocab_path = "data/scidocs_vocab.pkl"
        amazon_vocab_path = "data/amazon_vocab.pkl"
        french_vocab_path = "data/french_news_vocab.pkl"
        merged_vocab_path = "data/merged_vocab.pkl"

        scidocs_words_paths = [scidocs_bert_base_result_save_path, 
                               scidocs_scibert_result_save_path,
                               scidocs_flaubert_result_save_path]
        amazon_words_paths = [amazon_bert_base_result_save_path, 
                              amazon_scibert_result_save_path, 
                              amazon_flaubert_result_save_path] 
        french_words_paths = [french_bert_base_result_save_path,
                              french_scibert_result_save_path, 
                              french_flaubert_result_save_path]
        merged_words_paths = [merged_bert_base_result_save_path, 
                              merged_scibert_result_save_path, 
                              merged_flaubert_result_save_path]

        model_list = ['Bert Base', 'SciBert', 'FlauBert']


        # Decision paths
        scidocs_combined_decision_save_path = f"decision_data/scidocs_decision_combined_{criteria}.pkl"
        amazon_combined_decision_save_path = f"decision_data/amazon_decision_combined_{criteria}.pkl"
        french_combined_decision_save_path = f"decision_data/french_decision_combined_{criteria}.pkl"
        merged_combined_decision_save_path = f"decision_data/merged_decision_combined_{criteria}.pkl"
        
        # Doc info paths
        scidocs_doc_info_path = "data/scidocs_doc.pkl"
        amazon_doc_info_path = "data/amazon_doc.pkl"
        french_doc_info_path = "data/french_news_doc.pkl"
        merged_doc_info_path = "data/merged_doc.pkl"

        

        scidocs_combined = CombineCriteria(vocab_info_path=scidocs_vocab_path,
                                                doc_info_path=scidocs_doc_info_path,
                                                categorized_words_paths=scidocs_words_paths, 
                                                model_list=model_list,
                                                criteria=criteria, 
                                                top_k=top_k)
        scidocs_combined.combine_results()
        scidocs_combined.save_result(result_save_path=scidocs_combined_result_save_path,
                                          decision_save_path=scidocs_combined_decision_save_path)
        del scidocs_combined

        amazon_combined = CombineCriteria(vocab_info_path=amazon_vocab_path,
                                               doc_info_path=amazon_doc_info_path,
                                               categorized_words_paths=amazon_words_paths,
                                               model_list=model_list,
                                               criteria=criteria, 
                                               top_k=top_k)
        amazon_combined.combine_results()
        amazon_combined.save_result(result_save_path=amazon_combined_result_save_path,
                                         decision_save_path=amazon_combined_decision_save_path)
        del amazon_combined

        french_combined = CombineCriteria(vocab_info_path=french_vocab_path,
                                               doc_info_path=french_doc_info_path,
                                               categorized_words_paths=french_words_paths, 
                                               model_list=model_list,
                                               criteria=criteria, 
                                               top_k=top_k)
        french_combined.combine_results()
        french_combined.save_result(result_save_path=french_combined_result_save_path,
                                         decision_save_path=french_combined_decision_save_path)
        del french_combined

#        merged_combined = CombineCriteria(vocab_info_path=merged_vocab_path,
#                                               doc_info_path=merged_doc_info_path,
#                                               categorized_words_paths=merged_words_paths, 
#                                               model_list=model_list,
#                                               criteria=criteria, 
#                                               top_k=top_k)
#        merged_combined.combine_results()
#        merged_combined.save_result(result_save_path=merged_combined_result_save_path, 
#                                         decision_save_path=merged_combined_decision_save_path)
#        del merged_combined
#

        # Combied Scores Embeddings
        scidocs_scores_info_path = "data/scidocs_scores.pkl"
        amazon_scores_info_path = "data/amazon_scores.pkl"
        french_scores_info_path = "data/french_news_scores.pkl"
        merged_scores_info_path = "data/merged_scores.pkl"

        scidocs_combined_scores_embeds_save_path = f"result_data/scidocs_result_combined_scores_embeds_{criteria}.pkl"
        amazon_combined_scores_embeds_save_path = f"result_data/amazon_result_combined_scores_embeds_{criteria}.pkl"
        french_combined_scores_embeds_save_path = f"result_data/french_result_combined_scores_embeds_{criteria}.pkl"
        merged_combined_scores_embeds_save_path = f"result_data/merged_result_combined_scores_embeds_{criteria}.pkl"
        

        scidocs_combined_scores = CombineScoresEmbeddings(vocab_info_path=scidocs_vocab_path, scores_info_path=scidocs_scores_info_path,
                                                doc_info_path=scidocs_doc_info_path,
                                                categorized_words_path=scidocs_combined_result_save_path,
                                                criteria=criteria, 
                                                top_k=top_k)
        scidocs_combined_scores.combine_results()
        scidocs_combined_scores.save_result(result_save_path=scidocs_combined_scores_embeds_save_path)
        del scidocs_combined_scores

        amazon_combined_scores = CombineScoresEmbeddings(vocab_info_path=amazon_vocab_path, scores_info_path=amazon_scores_info_path,
                                               doc_info_path=amazon_doc_info_path,
                                               categorized_words_path=amazon_combined_result_save_path,
                                               criteria=criteria, 
                                               top_k=top_k)
        amazon_combined_scores.combine_results()
        amazon_combined_scores.save_result(result_save_path=amazon_combined_scores_embeds_save_path)
        del amazon_combined_scores

        french_combined_scores = CombineScoresEmbeddings(vocab_info_path=french_vocab_path, scores_info_path=french_scores_info_path,
                                               doc_info_path=french_doc_info_path,
                                               categorized_words_path=french_combined_result_save_path,
                                               criteria=criteria, 
                                               top_k=top_k)
        french_combined_scores.combine_results()
        french_combined_scores.save_result(result_save_path=french_combined_scores_embeds_save_path)
        del french_combined_scores

#        merged_combined_scores = CombineScoresEmbeddings(scores_info_path=merged_scores_info_path,
#                                               doc_info_path=merged_doc_info_path,
#                                               categorized_words_path=merged_combined_result_save_path,
#                                               criteria=criteria, 
#                                               top_k=top_k)
#        merged_combined_scores.combine_results()
#        merged_combined_scores.save_result(result_save_path=merged_combined_scores_embeds_save_path)
#        del merged_combined_scores
