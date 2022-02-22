"""
Creates a Dataset Wrapper for pytorch
@author: Philipp
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# Expects a pandas Dataframe with at least an ['embedding'] column, for training further a 'label_encoded' column is needed
# Split allows to optionally access a 'Split' column and only take those rows that equal to the input string
class CustomDataset(Dataset):
    def __init__(self, df_sentences, Split=False):
        if Split:
            self.df_sentences = df_sentences[df_sentences.Split == Split].reset_index(drop=True)
        else:
            self.df_sentences = df_sentences

    def __len__(self):
        return len(self.df_sentences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = torch.as_tensor(np.array(self.df_sentences.at[idx, 'embedding']), dtype=torch.float).unsqueeze(0)
        # If dataset is not used for training, no labels have to be passed
        if 'label_encoded' in self.df_sentences:
            label = self.df_sentences.at[idx, 'label_encoded']
            # This is so the label has the same dimension as network output/logits
            tmp = [0, 0, 0, 0, 0, 0]
            tmp[label] = 1
            label = (torch.tensor(tmp, dtype=torch.float))
            sample = {'sentence': sentence, 'index': idx, 'label': label}
        else:
            sample = {'sentence': sentence, 'index': idx}

        return sample
