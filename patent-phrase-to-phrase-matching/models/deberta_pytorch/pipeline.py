import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from torch import nn
from torchmetrics import MeanSquaredError
from tqdm import tqdm
from transformers import AutoTokenizer

from patent_phrase_similarity.data.transformation.cpc_datasets import CPCDatasets, Datasets
from patent_phrase_similarity.models.deberta_pytorch import CFG
from patent_phrase_similarity.models.deberta_pytorch.datasets import PPPMDatasetManager, CVSplitter, PPPMDataset
from patent_phrase_similarity.models.deberta_pytorch.model import PPPMTrainerDefinition, PPPMModel


class Pipeline:
    @staticmethod
    def calculate_dynamic_max_len(_df, columns_length, tokenizer):
        lengths_dict = {}
        lengths = []
        for column in columns_length:
            unique_values = list(_df[column].unique())
            tk0 = tqdm(unique_values, total=len(unique_values), desc=f"Calculating {column}")
            for val in tk0:
                if val is None:
                    val = ""
                length = len(tokenizer(val, add_special_tokens=False)['input_ids'])
                lengths.append(length)
            lengths_dict[column] = lengths

        # Sum of all tokens length
        # ([CLS] + [SEP] + [SEP] + [SEP]) = 4
        config = CFG()
        config.max_len = (sum(max(x) for x in lengths_dict.values()) + 4)
        print(f"max_len: {config.max_len}")

    @staticmethod
    def _load_data():
        config = CFG()

        datasets = Datasets()
        cpc_datasets = CPCDatasets()

        train_df = datasets.get_train_df()
        test_df = datasets.get_test_df()
        train_df = cpc_datasets.merge_with_df(train_df)
        test_df = cpc_datasets.merge_with_df(test_df)

        return train_df, test_df

    def pipeline(self):
        config = CFG()

        train_df, test_df = self._load_data()

        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.calculate_dynamic_max_len(train_df, ['anchor', 'target', 'title'], tokenizer)

        if config.folds >= 2:
            train_cv_splitter = CVSplitter(
                train_df, by="score", n_splits=config.folds, shuffle=True, seed=config.seed
            )

            train_df = train_cv_splitter.create_folds()

        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        metric = MeanSquaredError()
        model = PPPMModel(
            model_path=config.model.model_name,
            config_path=None,
            config_updates=None,
            reinitialization_layers=0,
            mixout=None
        )
        for kfold in config.fold_ids:
            if config.folds > 1:
                train_size = train_df.groupby('fold').size()[kfold]
            else:
                train_size = len(train_df)

            dataloader = PPPMDatasetManager(
                tokenizer, train_df, test_df, "score", kfold=kfold, validation_size=config.validation_size
            )
            trainer_definition = PPPMTrainerDefinition(
                model, criterion, metric, train_size
            )
            tqdm_progress_bar = TQDMProgressBar(refresh_rate=20)

            trainer = Trainer(
                max_epochs=config.epochs,
                accelerator="auto",
                devices=1 if config.device == "cuda" else None,  # limiting got iPython runs
                callbacks=[tqdm_progress_bar],
                logger=TensorBoardLogger('logs/', name=f"{config.model.model_name}_{kfold}"),
            )
            trainer.fit(
                trainer_definition, datamodule=dataloader
            )


if __name__ == '__main__':
    Pipeline().pipeline()





