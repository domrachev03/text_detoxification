import os
import datasets
import transformers
import pandas as pd


class ParaNMTDataset:
    def __init__(
            self,
            tokenizer,
            max_input_len = 256,
            download_dataset: bool = True,
            dataset_dir: str = 'data/raw'
    ):
        self._tokenizer = tokenizer
        self._max_len = max_input_len

        if download_dataset:
            self._download_dataset(dataset_dir)

    def _download_dataset(self, dir: str) -> None:
        filename = 'filtered_paranmt.zip'
        print(filename)
        os.system('wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip')
        os.system(f'unzip {filename} -d {dir}')
        os.system(f'rm {filename}')
    
    def load_dataset(self, file: str) -> datasets.Dataset:
        # Load the WMT16 dataset
        assert file.split('.')[-1] == '.tsv'

        self._df = pd.read_csv(file, sep='\t', index_col=0)
        self._df = self.preprocess_df()
        self._base_dataset = datasets.Dataset.from_pandas(self._df).remove_columns('__index_level_0__')
        self._prefix = "Make this text non-toxic:"
        return self._base_dataset

    def preprocess_df(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            new_df = self._df.copy()
        else:
            new_df = df.copy()

        cond = (new_df["ref_tox"] < new_df["trn_tox"])
        new_df.loc[cond, ['reference', 'translation', 'ref_tox', 'trn_tox']] = (
            new_df.loc[cond, ['translation', 'reference', 'trn_tox', 'ref_tox']].values)
        new_df = new_df[(new_df["ref_tox"] > 0.99) & (new_df["trn_tox"] < 0.01)]

        return new_df

    def _map_fn(self, prefix: str, examples: datasets.Dataset):
        inputs = [prefix + ref for ref in examples["reference"]]
        targets = [tsn for tsn in examples["translation"]]
        model_inputs = self.tokenizer(inputs, max_length=self._max_len, truncation=True, return_overflowing_tokens=False)

        # Setup the tokenizer for targets
        labels = self.tokenizer(targets, max_length=self._max_len, truncation=True, return_overflowing_tokens=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def preprocess_dataset(self, dataset: datasets.Dataset = None):
        pass
    
if __name__ == '__main__':
    dataset = ParaNMTDataset()
    dataset.download_dataset()