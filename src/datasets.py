from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class SciDocsDataset(Dataset):
    def __init__(self, scidocs_file_path, sample_ratio=1):
        self.scidocs_file = pd.read_csv(scidocs_file_path, na_filter=False).sample(frac=sample_ratio, ignore_index=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.scidocs_file)

    def __getitem__(self, idx):
        label = {"sentence_1": self.scidocs_file.loc[idx, "sentence_1"],
                 "sentence_2": self.scidocs_file.loc[idx, "sentence_2"],
                 "nsp": str(self.scidocs_file.loc[idx, 'nsp'])}
        return label
