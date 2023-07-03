import re

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, logging
import os

from src.rules import get_vocab_removal_rules

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# logging.set_verbosity_error()

save_512_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/result_data"

EMBED_SIZE = 768
BATCH_SIZE = 1000


def convert_seed_words_to_embeddings(word: str, add_special_tokens: bool):
    inputs = tokenizer(word, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state


def mean_seed_subwords_embeddings(subwords: torch.Tensor, add_special_tokens: bool):
    if add_special_tokens:
        return torch.mean(subwords[:, 1:-1, :], dim=1)
    else:
        return torch.mean(subwords, dim=1)


def get_seed_word_embeddings(seed_word: str, add_special_tokens: bool):
    subwords_embeddings = convert_seed_words_to_embeddings(seed_word, add_special_tokens=add_special_tokens)
    return convert_seed_words_to_embeddings(subwords=subwords_embeddings, add_special_tokens=add_special_tokens)


def get_subword_mask(input_ids: torch.Tensor):
    # TODO: Check shape
    subword_mask = torch.zeros(input_ids.flatten().shape).to(device)
    for wrd_idx in range(input_ids.flatten().shape[0]):
        if tokenizer.decode([input_ids.flatten()[wrd_idx]]).startswith("##"):
            subword_mask[wrd_idx] = 1
    return subword_mask


# TODO: Check mean properly
def get_word_embeddings_from_sentence(sentence: str, add_special_tokens: bool):
    tokenized_input_ids = tokenizer(sentence, add_special_tokens=add_special_tokens, return_tensors="pt",
                                    truncation=True).to(device)
    subword_mask = get_subword_mask(input_ids=tokenized_input_ids['input_ids'])
    result = torch.zeros(
        tokenized_input_ids['input_ids'].flatten().shape[0] - int(torch.min(torch.sum(subword_mask, dim=0)).item()),
        EMBED_SIZE).to(device)
    # print(tokenized_input_ids['input_ids'].flatten().shape)
    #print(subword_mask)
    #print(result.shape)
    output_embeddings = model(**tokenized_input_ids).last_hidden_state
    #print(output_embeddings)

    mean_embeddings = mean_count = idx_offset = 0
    for wrd_idx in range(output_embeddings.shape[1]):
        if subword_mask[wrd_idx]:
            mean_embeddings += output_embeddings[0, wrd_idx]
            mean_count += 1
            idx_offset += 1
            if not subword_mask[wrd_idx - 1]:
                mean_embeddings += output_embeddings[0, wrd_idx - 1]
                mean_count += 1
            if wrd_idx == output_embeddings.shape[1] - 1 or not subword_mask[wrd_idx + 1]:
                result[wrd_idx - idx_offset] = mean_embeddings / mean_count
                mean_embeddings = 0
                mean_count = 0
        else:
            result[wrd_idx - idx_offset] = output_embeddings[0, wrd_idx]
    #print(sentence)
    #print(tokenized_input_ids['input_ids'])
    return result[1:-1, :], tokenized_input_ids['input_ids'].shape[1]


def process_sentences(sentences_df: pd.DataFrame):
    result_dict = {}
    sentences_dict = sentences_df.to_dict('records')
    for count, row in enumerate(sentences_dict):
        print("Processing Sentences ... [{curr} / {total}]".format(curr=str(count + 1), total=str(len(sentences_dict))),
              end='\r')
        # TODO: Too NAIVE
        sentence = re.sub(r'[^\w\s\'-]|\d', ' ', row['sentences']).strip()
        sentence_embeddings, tensor_size = get_word_embeddings_from_sentence(sentence=sentence, add_special_tokens=True)
        for idx, word in enumerate(list(filter(None, sentence.strip().split()))):
            try:
                if word not in result_dict:
                    result_dict[word] = [sentence_embeddings[idx].cpu().detach().numpy()]
                else:
                    result_dict[word].append(sentence_embeddings[idx].cpu().detach().numpy())
            except IndexError:
                if tensor_size == 512:
                    continue
                else:
                    raise IndexError
        if count == len(sentences_dict):
            #print(result_dict)
            #print(pd.DataFrame(result_dict))
            #data = {key: [array for array in value] for key, value in result_dict.items()}
            pd.DataFrame(list(result_dict.items()), columns=['words', 'embeddings']).to_pickle(
                os.path.join(save_512_path, f"vocab_tensors_no_nsp_no_punctuations_keep_dot.csv"))
            #result_dict = {}
        count += 1


def calculate_shortest_distance():
    pass


if __name__ == "__main__":
    dataset_512_path = "/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/scidocs_data/scidocs_dataset_512_raw.csv"
    # TODO: Check NA Filter
    dataset_512_df = pd.read_csv(dataset_512_path, na_filter=False)#.sample(n=1000000, random_state=1)
    process_sentences(sentences_df=dataset_512_df)
