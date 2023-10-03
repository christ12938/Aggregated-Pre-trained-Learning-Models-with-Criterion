import os
import regex as re
from tqdm import tqdm
import pandas as pd


def clean_sentence(sentence: str, regex_rules: str):
    cleaned_sentence = re.sub(regex_rules, '', sentence)
    cleaned_sentence = ' '.join(cleaned_sentence.split())
    return cleaned_sentence


def create_vocab_info_df(sentences_list: list, id_prefix: str):
    vocab_info_dict = {}
    for idx, sentence in enumerate(tqdm(sentences_list, desc="Creating Vocabulary")):
        clean_vocabs = list(filter(None, sentence.split()))
        for vocab in clean_vocabs:
            vocab_info_dict.setdefault(vocab, set())
            vocab_info_dict[vocab].add(f"{id_prefix}_{idx}")
    vocab_info_df = pd.DataFrame(list(vocab_info_dict.items()), columns=['vocab', 'id'])
    return vocab_info_df


def create_folder(path: str):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
