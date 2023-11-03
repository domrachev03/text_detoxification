import os
import datasets
import pandas as pd
import argparse
from collections.abc import Iterable


class DetoxDataset:
    '''Class DetoxDataset.

    Serves as a convinient proxy for loading, preprocessing and saving the dataset locally and to the cloud.
    Suggested use: only once, at the beginning, to initialize dataset.'''

    def __init__(
        self,
        tokenizer=None,                         
        max_tokens=256,
        wrapper=lambda x: x,
        lazy_init: bool = False,
        download_dataset: bool = True,
        load_and_preprocess: bool = True,
        split_and_map_dataset: bool = False,
        split_dataset: bool = True,
        save_splitted: bool = True,
        dataset_sv_fname: str = 'data/raw/filtered.tsv'
    ):
        '''Default behaviour at construction: download, preprocess and save the dataset locally, without tokenization'''

        self._tokenizer = tokenizer
        self._wrapper = wrapper
        self._max_tokens = max_tokens

        if lazy_init:
            return

        if download_dataset:
            dataset_dir = os.path.dirname(dataset_sv_fname)
            self._download_dataset(dataset_dir)
        if load_and_preprocess:
            self.load_from_sv(dataset_sv_fname)
            self.preprocess_dataset_sv()
        if split_and_map_dataset:
            self.preprocess_dataset(save_arrow=save_splitted)
        elif split_dataset:
            self.split_dataset(save_arrow=save_splitted)

    def set_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def dataset(self):
        return self._dataset

    def _download_dataset(self, dir: str, zip_name: str = 'filtered_paranmt') -> None:
        ''' Downloading and unzipping dataset from the Internet.
            Requires unzip package installed on Linux system '''

        os.system('wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/{zip_name}')
        os.system(f'unzip {zip_name} -d {dir}')
        os.system(f'rm {zip_name}')

    def load_from_sv(self, file: str = "data/raw/filtered.tsv") -> datasets.Dataset:
        '''Loadinf the dataset from the tsv file and putting it into the datasets.Dataset'''

        assert file.split('.')[-1] == 'tsv'

        self._df = pd.read_csv(file, sep='\t', index_col=0)
        self._df = self.preprocess_dataset_sv()
        self._dataset = datasets.Dataset.from_pandas(self._df).remove_columns('__index_level_0__')
        return self._dataset

    def preprocess_dataset_sv(self, df: pd.DataFrame = None) -> pd.DataFrame:
        '''Preprocessing of the dataset in the tsv format'''

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

    def _map_fn(self, examples: datasets.Dataset) -> dataset.Datasets:
        ''' Mapping function, used to tokenize the dataset '''

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is undefined")
        inputs = [self._wrapper(ref) for ref in examples["reference"]]
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
            load_whole_ds=True,
            n_samples: Iterable[int] = (5000, 500),
            test_size: float = 0.1,
            batch_size: float = 256,
            save_arrow: bool = False,
            fdir: str = 'data/interim'
    ) -> datasets.Dataset:
        '''Splitting the dataset onto train and test parts.

        Parameters:
        load_whole_ds (bool): if False, utilized proportions from the n_samples to cut the datasets

        test_size (float): the fraction of the data utilized as test dataset, ranges [0; 1]

        save_arrow (bool): if True, then the dataset would be saved to the file
        '''

        if df is not None:
            self._dataset = datasets.Dataset.from_pandas(df).remove_columns('__index_level_0__')

        split_dict = self._dataset.train_test_split(
            test_size=test_size,
            seed=42,
        )
        batch_size = batch_size
        if not load_whole_ds:
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
            load_whole_ds=True,
            n_samples: Iterable[int] = (5000, 500),
            test_size=0.1,
            batch_size=256,
            save_arrow=False,
            fdir='data/interim'
    ) -> datasets.Dataset:
        ''' Preprocessing the data, i.e. tokenizing and splitting.

        Parameters: see split_dataset() '''
        split_dict = self.split_dataset(
            df,
            load_whole_ds=load_whole_ds,
            n_samples=n_samples,
            test_size=test_size,
            batch_size=batch_size,
            save_arrow=save_arrow,
            fdir=fdir
        )

        tokenized_datasets = split_dict.map(
            self._map_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=split_dict["train"].column_names
        )
        self._dataset = tokenized_datasets

        return tokenized_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--push", help="if true, pushes the dataset to huggingface.co. Default: false", type=bool, default=False)
    parser.add_argument('-d', "--download", help="if true, downloads the zip archive with raw data. Default: false", type=bool, default=False)

    args = parser.parse_args()

    # Creates a copy of the dataset and saves it locally.
    ds = DetoxDataset(
        tokenizer=None,
        download_dataset=args.download,
        save_splitted=True
    )

    if args.push:
        ds.push_to_hub('domrachev03/toxic_comments_subset')
