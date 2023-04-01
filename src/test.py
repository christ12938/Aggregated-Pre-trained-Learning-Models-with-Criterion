from datasets import SciDocsDataset
import pandas as pd
from torch.utils.data import DataLoader
from custom_bert import CustomBert

scidocs_data = SciDocsDataset("/home/chris/thesis/src/data/scidocs_data/scidocs_dataset_128.csv")


custom_bert = CustomBert(dataset=scidocs_data)
custom_bert.train_one_epoch()
#scidocs_data = SciDocsDataset("/home/chris/thesis/src/data/scidocs_data/scidocs_dataset_128.csv")
#print(scidocs_data)
#bert_training = CustomBertMethod(dataset=scidocs_data)
#bert_training.train_one_epoch(epoch=1)
"""
epoch = 10
scidocs_data = SciDocsDataset("scidocs_dataset_128.csv", sample_ratio=0.2)
bert_training = BertFineTuneMethod(scidocs_data, epoch=10, max_len=128)
for i in range(1, epoch):
    bert_training.train_one_epoch(epoch=i)
"""