import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

bert_base_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_base_model = AutoModel.from_pretrained('bert-base-uncased').to(device)

scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)


def convert_words_to_embeddings(word: str, add_special_tokens: bool, model_options: str):
    if model_options == "bert_base":
        inputs = bert_base_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        outputs = bert_base_model(**inputs)
        return outputs.last_hidden_state
    elif model_options == "scibert":
        inputs = scibert_tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        outputs = scibert_model(**inputs)
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
            pd.read_csv(self.vocab_path, na_filter=False).loc[:, "vocab"])  # .sample(n=1000000, random_state=1)
        result_dict = {key: [] for key in vocab_list}
        for vocab in tqdm(vocab_list, desc="Processing Vocabs"):
            result_dict[vocab].append(
                get_word_embeddings(seed_word=vocab.strip(), add_special_tokens=True,
                                    model_options=self.model_options).cpu().detach().numpy())
        self.vocab_embed_dict = result_dict

    def save_vocab_embeddings(self, save_path: str):
        print("Saving Vocab Embeddings ...")
        pd.DataFrame(list(self.vocab_embed_dict.items()), columns=['vocabs', 'embeddings']).to_pickle(save_path)


if __name__ == "__main__":
    vocab_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_vocab_no_punc_no_special_char_keep_apos_hyphens.csv"

    vocab_embed_bert_base_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_bert_base.pkl"
    vocab_embed_scibert_save_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data/vocab_tensors_scibert.pkl"

    bert_base_vocab_embeds = VocabEmbeddings(vocab_path=vocab_path, model_options="bert_base")
    bert_base_vocab_embeds.process_vocabs()
    bert_base_vocab_embeds.save_vocab_embeddings(save_path=vocab_embed_bert_base_save_path)
    del bert_base_vocab_embeds

    scibert_vocab_embeds = VocabEmbeddings(vocab_path=vocab_path, model_options="scibert")
    scibert_vocab_embeds.process_vocabs()
    scibert_vocab_embeds.save_vocab_embeddings(save_path=vocab_embed_scibert_save_path)
    del scibert_vocab_embeds
