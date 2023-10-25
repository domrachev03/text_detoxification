import os
import datasets
import pandas as pd
from collections.abc import Iterable
from llama_recipes.datasets.utils import Concatenator


class ParaNMTDataset:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=256,
        prefix="You are a Twitch moderator. Make this text non-toxic: \n",
        download_dataset: bool = False,
        map_dataset: bool = False,
        save_splitted: bool = False,
        dataset_sv_fname: str = 'data/raw/filtered.tsv'
    ):
        self._tokenizer = tokenizer
        self._prefix = prefix
        self._max_tokens = max_tokens

        if download_dataset:
            dataset_dir = os.path.dirname(dataset_sv_fname)
            self._download_dataset(dataset_dir)
        
        self.load_from_sv(dataset_sv_fname)
        self.preprocess_dataset_sv()
        if map_dataset:
            self.preprocess_dataset()
        else:
            self.split_dataset(save_arrow=save_splitted)

    def set_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def dataset(self):
        return self._dataset

    def _download_dataset(self, dir: str, zip_name: str = 'filtered_paranmt') -> None:
        os.system('wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/{zip_name}')
        os.system(f'unzip {zip_name} -d {dir}')
        os.system(f'rm {zip_name}')

    def load_from_sv(self, file: str = "data/raw/filtered.tsv") -> datasets.Dataset:
        # Load the WMT16 dataset
        assert file.split('.')[-1] == 'tsv'

        self._df = pd.read_csv(file, sep='\t', index_col=0)
        self._df = self.preprocess_dataset_sv()
        self._dataset = datasets.Dataset.from_pandas(self._df).remove_columns('__index_level_0__')
        return self._dataset

    def preprocess_dataset_sv(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            new_df = self._df.copy()
        else:
            new_df = df.copy()

        cond = (new_df["ref_tox"] < new_df["trn_tox"])
        new_df.loc[cond, ['reference', 'translation', 'ref_tox', 'trn_tox']] = (
            new_df.loc[cond, ['translation', 'reference', 'trn_tox', 'ref_tox']].values)
        new_df = new_df[(new_df["ref_tox"] > 0.99) & (new_df["trn_tox"] < 0.01)]
        self._df = new_df

        return new_df

    def _map_fn(self, examples: datasets.Dataset):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is undefined")
        inputs = [self._prefix + ref for ref in examples["reference"]]
        targets = [tsn for tsn in examples["translation"]]
        model_inputs = self._tokenizer(
            inputs,
            max_length=self._max_tokens,
            truncation=True,
            return_overflowing_tokens=False
        )

        # Setup the tokenizer for targets
        labels = self._tokenizer(
            targets,
            max_length=self._max_tokens,
            truncation=True,
            return_overflowing_tokens=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def split_dataset(
            self,
            df: pd.DataFrame = None,
            n_samples: Iterable[int] = (5000, 500),
            test_size=0.1,
            batch_size=256,
            save_arrow=False,
            fdir='data/interim'
    ) -> datasets.Dataset:

        if df is not None:
            self._dataset = datasets.Dataset.from_pandas(df).remove_columns('__index_level_0__')

        split_dict = self._dataset.train_test_split(
            test_size=test_size,
            seed=42,
        )
        batch_size = batch_size
        split_dict['train'] = split_dict['train'].select(range(n_samples[0]))
        split_dict['test'] = split_dict['test'].select(range(n_samples[1]))
        if save_arrow:
            split_dict.push_to_hub('domrachev03/toxic_comments_subset')
            split_dict["train"].save_to_disk(fdir + '/train_unmapped')
            split_dict["test"].save_to_disk(fdir + '/test_unmapped')

        return split_dict

    def preprocess_dataset(
            self,
            df: pd.DataFrame = None,
            n_samples: Iterable[int] = (5000, 500),
            test_size=0.1,
            batch_size=256,
            save_arrow=False,
            fdir='data/interim'
    ) -> datasets.Dataset:
        split_dict = self.split_dataset(df, n_samples, test_size, batch_size, False, fdir)

        tokenized_datasets = split_dict.map(
            self._map_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=split_dict["train"].column_names
        )
        self._dataset = tokenized_datasets
        if save_arrow:
            tokenized_datasets["train"].save_to_disk(fdir + '/train')
            tokenized_datasets["test"].save_to_disk(fdir + '/test')

        return tokenized_datasets


if __name__ == '__main__':
    ds = ParaNMTDataset(
        tokenizer=None,
        download_dataset=False,
        save_splitted=True
    )
