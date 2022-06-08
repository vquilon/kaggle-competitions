import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import AutoTokenizer

from patent_phrase_similarity.models.deberta_pytorch import CFG


class PPPMDataset(Dataset):
    def __init__(self, df, tokenizer, for_training=True):
        self.config = CFG()
        self.df = df
        self.tokenizer = tokenizer
        self.training = for_training
        if for_training:
            self.labels = df.score.values

    def __len__(self):
        if self.training:
            return len(self.df)
        else:
            return len(self.labels)

    def __getitem__(self, index):
        anchor = self.df.anchor.iloc[index].lower()
        target = self.df.target.iloc[index].lower()
        title = self.df.title.iloc[index].lower()

        SEP = self.tokenizer.sep_token
        tokens = self.tokenizer(
            f"{anchor}{SEP}{target}{SEP}{title}",
            max_length=self.config.max_len,
            add_special_tokens=True,
            padding="max_length",
            # truncation=True,
            return_attention_mask=True,
            return_token_type_ids=self.config.token_type_ids,
            return_offsets_mapping=False
        )
        for k, v in tokens.items():
            tokens[k] = torch.tensor(v, dtype=torch.long)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.training:
            return tokens['input_ids'], tokens['attention_mask'], label

        return tokens['input_ids'], tokens['attention_mask']


class CustomDataset(Dataset):
    def __init__(self, texts, pair_texts, tokenizer, contexts=None, sep=None, targets=None, max_length=128):
        self.config = CFG()
        self.texts = texts
        self.pair_texts = pair_texts
        self.contexts = contexts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep = sep if sep is not None else self.tokenizer.sep_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index].lower()
        pair_text = self.pair_texts[index].lower()

        if self.contexts is not None:
            context = self.contexts[index].lower()
            text = text + self.sep + context

        tokenized = self.tokenizer(
            text=text,
            text_pair=pair_text,
            add_special_tokens=True,
            #max_length=self.max_length,
            #padding="max_length",
            #truncation=True,
            return_attention_mask=True,
            return_token_type_ids=self.config.token_type_ids,
            return_offsets_mapping=False
        )

        if self.targets is not None:
            target = self.targets[index]
            return tokenized, target

        return tokenized


class DynamicPadding:
    def __init__(self, tokenizer, max_length=None, padding=True, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, tokenized):
        max_length = max(len(_["input_ids"]) for _ in tokenized)
        max_length = min(max_length, self.max_length) if self.max_length is not None else max_length

        padded = self.tokenizer.pad(
            encoded_inputs=tokenized,
            max_length=max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        return padded


class Collator:
    def __init__(self, return_targets=True, **kwargs):
        self.dynamic_padding = DynamicPadding(**kwargs)
        self.return_targets = return_targets

    def __call__(self, batch):
        config = CFG()

        all_tokenized, all_targets = [], []
        for sample in batch:
            if self.return_targets:
                tokenized, target = sample
                all_targets.append(target)
            else:
                tokenized = sample

            all_tokenized.append(tokenized)

        tokenized = self.dynamic_padding(all_tokenized)

        input_ids = torch.tensor(tokenized.input_ids)
        attention_mask = torch.tensor(tokenized.attention_mask)
        token_type_ids = torch.tensor(tokenized.token_type_ids) if config.token_type_ids else None

        if self.return_targets:
            all_targets = torch.tensor(all_targets)

            if token_type_ids is not None:
                return input_ids, attention_mask, token_type_ids, all_targets

            return input_ids, attention_mask, all_targets
        else:
            if token_type_ids is not None:
                return input_ids, attention_mask, token_type_ids

        return input_ids, attention_mask


class CVSplitter:
    FOLD_COLUMN = "fold"

    def __init__(self, _df, by="", n_splits=0, shuffle=True, seed=None, debug=False):
        self.column_fold = self.FOLD_COLUMN
        self._df = _df
        self._by = by
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.debug = debug

    def create_folds(self, _df=None, by="", n_splits=0, shuffle=True, seed=None) -> pd.DataFrame:
        if _df is None:
            _df = self._df
        if by == "":
            by = self._by
        if n_splits == 0:
            n_splits = self.n_splits
        if shuffle is None:
            shuffle = self.shuffle
        if seed is None:
            seed = self.seed

        _Fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        for n, (train_index, val_index) in enumerate(_Fold.split(_df, _df[by])):
            _df.loc[val_index, self.column_fold] = int(n)

        _df[self.column_fold] = _df[self.column_fold].astype(int)
        if self.debug:
            print(_df.groupby(self.column_fold).size())
        return _df


class PPPMDatasetManager(pl.LightningDataModule):
    def __init__(self, tokenizer, train_df, test_df, label_column, fold_column=CVSplitter.FOLD_COLUMN, kfold=None, validation_size=0.2):
        super(PPPMDatasetManager, self).__init__()
        self.config = CFG()

        self.tokenizer = tokenizer
        if kfold is not None and self.config.folds > 1:
            self.train_oof_df = train_df[train_df[fold_column] != kfold].reset_index(drop=True)
            self.valid_oof_df = train_df[train_df[fold_column] == kfold].reset_index(drop=True)
            self.train_labels = self.train_oof_df[label_column].values
            self.valid_labels = self.valid_oof_df[label_column].values
        else:
            self.train_labels = train_df[label_column].values
            self.train_oof_df, self.valid_oof_df, self.train_labels, self.valid_labels = train_test_split(
                train_df, self.train_labels,
                stratify=self.train_labels,
                test_size=validation_size,
                random_state=self.config.seed
            )
            self.valid_labels = train_df[label_column].values

        self.test_df = test_df

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.train_dataset = PPPMDataset(self.train_oof_df, self.tokenizer)
        self.valid_dataset = PPPMDataset(self.valid_oof_df, self.tokenizer)
        self.test_dataset = PPPMDataset(self.test_df, self.tokenizer, for_training=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
            pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, shuffle=False,
            pin_memory=True, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
            pin_memory=True, drop_last=False
        )
