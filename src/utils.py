import os
import regex as re
from tqdm import tqdm
import pandas as pd
import random


MAX_WORD_LIMIT = 510
CRITERIA_LIST = ['pmi_laplace', 
                 'ppmi', 'ppmi_delta', 'ppmi_laplace', 
                 'npmi', 'npmi_laplace', 
                 'wappmi_alpha_1', 'wappmi_alpha_1_delta', 'wappmi_alpha_1_laplace',
                 'wappmi_alpha_2', 'wappmi_alpha_2_delta', 'wappmi_alpha_2_laplace',
                 'wappmi_alpha_3', 'wappmi_alpha_3_delta', 'wappmi_alpha_3_laplace']


def clean_sentence(sentence: str, regex_rules: str):
    cleaned_sentence = re.sub(regex_rules, '', sentence)
    cleaned_sentence = ' '.join(cleaned_sentence.split())
    return cleaned_sentence


def create_vocab_info_df(sentences_list: list, id_prefix: str, sample: int):
    vocab_info_dict = {}
    vocab_count_dict = {}
    for idx, sentence in enumerate(tqdm(sentences_list, desc="Creating Vocabulary")):
        clean_vocabs = list(filter(None, sentence.split()))
        for vocab in clean_vocabs:
            if len(vocab) > MAX_WORD_LIMIT:
                continue
            doc_id = f"{id_prefix}_{idx}"
            vocab_info_dict.setdefault(vocab, dict())
            vocab_info_dict[vocab].setdefault(doc_id, 0)
            vocab_info_dict[vocab][doc_id] += 1
            vocab_count_dict.setdefault(vocab, 0)
            vocab_count_dict[vocab] += 1
    assert list(vocab_info_dict.keys()) == list(vocab_count_dict.keys())
    vocab_info_df = pd.DataFrame({'vocab': vocab_info_dict.keys(), 'id': vocab_info_dict.values(), 'count': vocab_count_dict.values()})
    vocab_info_df = vocab_info_df.sample(frac=sample).reset_index(drop=True)
    return vocab_info_df


def create_doc_info_df(vocab_info_df: pd.DataFrame):
    doc_info_dict = {}
    for entry in vocab_info_df['id']:
        for doc_id, count in entry.items():
            doc_info_dict.setdefault(doc_id, 0)
            doc_info_dict[doc_id] += count
    doc_info_df = pd.DataFrame(list(doc_info_dict.items()), columns=['id', 'length'])
    return doc_info_df


def create_folder(path: str):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def save_info(dataset: str, vocab_info_save_path: str, doc_info_save_path: str, vocab_info_df: pd.DataFrame, doc_info_df: pd.DataFrame):
    print(f'\nSaving {dataset} Vocab Info ... ')
    create_folder(path=vocab_info_save_path)
    create_folder(path=doc_info_save_path)
    vocab_info_df.to_pickle(vocab_info_save_path)
    doc_info_df.to_pickle(doc_info_save_path)
