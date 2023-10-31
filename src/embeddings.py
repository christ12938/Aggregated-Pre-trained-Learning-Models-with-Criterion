import sys

import pandas as pd
import torch
from utils import create_folder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

bert_base_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
bert_base_model = AutoModel.from_pretrained('bert-large-uncased').to(device)

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)

flaubert_tokenizer = AutoTokenizer.from_pretrained('flaubert/flaubert_large_cased')
flaubert_model = AutoModel.from_pretrained('flaubert/flaubert_large_cased').to(device)

#legalbert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
#legalbert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(device)


def convert_words_to_embeddings(word: str, add_special_tokens: bool, model_options: str):
    if model_options == "bert_base":
        inputs = bert_base_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        outputs = bert_base_model(**inputs)
        return outputs.last_hidden_state
    elif model_options == "scibert":
        inputs = scibert_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        outputs = scibert_model(**inputs)
        return outputs.last_hidden_state
    elif model_options == "legalbert":
        #inputs = legalbert_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        #outputs = legalbert_model(**inputs)
        #return outputs.last_hidden_state
        return None
    elif model_options == "flaubert":
        inputs = flaubert_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        outputs = flaubert_model(**inputs)
        return outputs.last_hidden_state

def mean_subwords_embeddings(subwords: torch.Tensor, add_special_tokens: bool):
    if add_special_tokens:
        return torch.mean(subwords[:, 1:-1, :], dim=1)
    else:
        return torch.mean(subwords, dim=1)


def get_word_embeddings(seed_word: str, add_special_tokens: bool, model_options: str):
    subwords_embeddings = convert_words_to_embeddings(seed_word, add_special_tokens=add_special_tokens,
                                                      model_options=model_options)
    return mean_subwords_embeddings(subwords=subwords_embeddings, add_special_tokens=add_special_tokens)


class VocabEmbeddings:
    def __init__(self, vocab_path: str, model_options: str):
        self.vocab_path = vocab_path
        self.vocab_embed_dict = None
        self.model_options = model_options

    def process_vocabs(self):
        vocab_list = list(
            pd.read_pickle(self.vocab_path).loc[:, "vocab"])  # .sample(n=1000000, random_state=1)
        result_dict = {}
        for vocab in tqdm(vocab_list, desc="Processing Vocabs"):
            result_dict[vocab] = get_word_embeddings(seed_word=vocab.strip(), add_special_tokens=True,
                                                     model_options=self.model_options).cpu().detach().numpy()
        self.vocab_embed_dict = result_dict

    def save_vocab_embeddings(self, save_path: str):
        print("Saving Vocab Embeddings ...")
        create_folder(path=save_path)
        pd.DataFrame(list(self.vocab_embed_dict.items()), columns=['vocab', 'embedding']).to_pickle(save_path)


class SeedEmbeddings:
    def __init__(self, seeds_list: list, model_options: str):
        self.seeds_list = seeds_list
        self.seed_embed_dict = None
        self.model_options = model_options

    def process_seeds(self):
        result_dict = {}
        for seed in tqdm(self.seeds_list, desc="Processing Seeds"):
            result_dict[seed] = get_word_embeddings(seed_word=seed.strip(), add_special_tokens=True,
                                                    model_options=self.model_options).cpu().detach().numpy()
        self.seed_embed_dict = result_dict

    def save_seed_embeddings(self, save_path: str):
        sys.stderr.flush()
        print("Saving Seed Embeddings ...")
        create_folder(path=save_path)
        pd.DataFrame(list(self.seed_embed_dict.items()), columns=['seed', 'embedding']).to_pickle(save_path)


if __name__ == "__main__":

    scidocs_vocab_path = "data/scidocs_vocab.pkl"
    amazon_vocab_path = "data/amazon_vocab.pkl"
    french_vocab_path = "data/french_news_vocab.pkl"
    merged_vocab_path = "data/merged_vocab.pkl"
    

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
   

    bert_base_vocab_embeds = VocabEmbeddings(vocab_path=scidocs_vocab_path, model_options="bert_base")
    bert_base_vocab_embeds.process_vocabs()
    bert_base_vocab_embeds.save_vocab_embeddings(save_path=scidocs_vocab_embed_bert_base_save_path)
    del bert_base_vocab_embeds

    bert_base_vocab_embeds = VocabEmbeddings(vocab_path=amazon_vocab_path, model_options="bert_base")
    bert_base_vocab_embeds.process_vocabs()
    bert_base_vocab_embeds.save_vocab_embeddings(save_path=amazon_vocab_embed_bert_base_save_path)
    del bert_base_vocab_embeds

    bert_base_vocab_embeds = VocabEmbeddings(vocab_path=french_vocab_path, model_options="bert_base")
    bert_base_vocab_embeds.process_vocabs()
    bert_base_vocab_embeds.save_vocab_embeddings(save_path=french_vocab_embed_bert_base_save_path)
    del bert_base_vocab_embeds

    bert_base_vocab_embeds = VocabEmbeddings(vocab_path=merged_vocab_path, model_options="bert_base")
    bert_base_vocab_embeds.process_vocabs()
    bert_base_vocab_embeds.save_vocab_embeddings(save_path=merged_vocab_embed_bert_base_save_path)
    del bert_base_vocab_embeds

    scibert_vocab_embeds = VocabEmbeddings(vocab_path=scidocs_vocab_path, model_options="scibert")
    scibert_vocab_embeds.process_vocabs()
    scibert_vocab_embeds.save_vocab_embeddings(save_path=scidocs_vocab_embed_scibert_save_path)
    del scibert_vocab_embeds

    scibert_vocab_embeds = VocabEmbeddings(vocab_path=amazon_vocab_path, model_options="scibert")
    scibert_vocab_embeds.process_vocabs()
    scibert_vocab_embeds.save_vocab_embeddings(save_path=amazon_vocab_embed_scibert_save_path)
    del scibert_vocab_embeds

    scibert_vocab_embeds = VocabEmbeddings(vocab_path=french_vocab_path, model_options="scibert")
    scibert_vocab_embeds.process_vocabs()
    scibert_vocab_embeds.save_vocab_embeddings(save_path=french_vocab_embed_scibert_save_path)
    del scibert_vocab_embeds

    scibert_vocab_embeds = VocabEmbeddings(vocab_path=merged_vocab_path, model_options="scibert")
    scibert_vocab_embeds.process_vocabs()
    scibert_vocab_embeds.save_vocab_embeddings(save_path=merged_vocab_embed_scibert_save_path)
    del scibert_vocab_embeds

    flaubert_vocab_embeds = VocabEmbeddings(vocab_path=scidocs_vocab_path, model_options="flaubert")
    flaubert_vocab_embeds.process_vocabs()
    flaubert_vocab_embeds.save_vocab_embeddings(save_path=scidocs_vocab_embed_flaubert_save_path)
    del flaubert_vocab_embeds

    flaubert_vocab_embeds = VocabEmbeddings(vocab_path=amazon_vocab_path, model_options="flaubert")
    flaubert_vocab_embeds.process_vocabs()
    flaubert_vocab_embeds.save_vocab_embeddings(save_path=amazon_vocab_embed_flaubert_save_path)
    del flaubert_vocab_embeds

    flaubert_vocab_embeds = VocabEmbeddings(vocab_path=french_vocab_path, model_options="flaubert")
    flaubert_vocab_embeds.process_vocabs()
    flaubert_vocab_embeds.save_vocab_embeddings(save_path=french_vocab_embed_flaubert_save_path)
    del flaubert_vocab_embeds

    flaubert_vocab_embeds = VocabEmbeddings(vocab_path=merged_vocab_path, model_options="flaubert")
    flaubert_vocab_embeds.process_vocabs()
    flaubert_vocab_embeds.save_vocab_embeddings(save_path=merged_vocab_embed_flaubert_save_path)
    del flaubert_vocab_embeds

