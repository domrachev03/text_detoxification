# Necessary inputs
import warnings

import datasets
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
                         Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import tqdm
import argparse

from src.metrics.metrics import compute_metrics

tqdm.pandas()
warnings.filterwarnings('ignore')
torch.manual_seed(42)


def train(
    dataset: datasets.dataset,
    tokenizer: datasets.tokenizer = None,
    n_epoches: int = 10,
    compute_metrics=compute_metrics,
    model_name: str = 't5-small',
    batch_size: int = 128,
    device: str = None,
    training_args: Seq2SeqTrainingArguments = None,
    save_model: bool = True,
    model_fname: str = 't5_best'
):
    ''' Function for training T5 (or other similar seq2seq) Neural Networks.
    Parameters:
    dataset (datasets.dataset) -- the training dataset. Expected to be generated via default procedure for
    huggingface datasets. Expected to have 'train' and 'val' parts.

    tokenizer (datasets.tokenizer) -- the tokenizer which was used for dataset creation.
    Default: AutoTokenizer.from_pretrained(model_name)

    n_epoches (int): no. of epoches to train. Default: 10.

    compute_metrics (function): function to compute metrics on the evaluation data.
    Default: function from metrics/metrics.py

    model_name (str): name of both model and tokenizer. Default: t5-small.

    batch_size (int). Default: 128

    device (cuda). Default: cuda if available, else cpu

    training_args (datasets.Seq2SeqTrainingArguments)

    save_mode (bool). Default: True

    model_fname (str): path to save the model weights locally. Default: "t5_best"
    '''

    t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name = model_name.split("/")[-1]

    if training_args is None:
        training_args = Seq2SeqTrainingArguments(
            f"{model_name}-finetuned-detoxification",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=10,
            num_train_epochs=10,
            predict_with_generate=True,
            fp16=True,
            report_to='tensorboard',
        )

    t5_data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5_model)
    t5_trainer = Seq2SeqTrainer(
        t5_model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=t5_data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, batch_size=batch_size)
    )

    t5_trainer.train()
    if save_model:
        t5_trainer.save_model(model_fname)

    return t5_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='t5-finetuner',
        description='Finetuning of the T5-small model on the given dataset. The results would be saved into specified directory'
    )
    parser.add_argument('-d', "--dataset", help="Path to the dataset", required=False, default='domrachev03/toxc_comments_subset')
    parser.add_argement('--n-train-samples', help="Number of elements in train set", type=int, default=1000)
    parser.add_argement('--n-test-samples', help="Number of elements in the test set", type=int, default=1000)
    parser.add_argument("--n-epoch", help='Number of epoches to learn model for. Default is 1.', type=int, default=1)
    parser.add_argument("--mode_name", help='Name of the model to finetune. Default is "t5-small"', type=str, default="t5-small")
    parser.add_argument("--batch-size", help='The size of the learning batch. Default is 128', type=int, default=128)
    parser.add_argument("--device", help='Device for computation, either "cuda" or "cpu"', type=str, default=None)
    parser.add_argument("--model-fname", help='Full name of the saved model', type=str, default="t5-best")

    args = parser.parse_args()

    '''
        dataset: datasets.dataset,
        tokenizer: datasets.tokenizer = None,
        n_epoches: int = 10,
        compute_metrics=compute_metrics,
        model_name: str = 't5-small',
        batch_size: int = 128,
        device: str = None,
        training_args: Seq2SeqTrainingArguments = None,
        save_model: bool = True,
        model_fname: str = 't5_best'
    '''
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = datasets.load_dataset(args.dataset)

    n_train = 50000
    n_val = 1000
    ds_subset = dataset['train'].select(range(n_train + n_val))

    dataset['train'] = ds_subset.select(range(n_train)) 
    dataset['val'] = ds_subset.select(range(n_train, n_train+n_val))

    train(
        dataset=dataset,
        tokenizer=tokenizer,
        n_epoches=args.n_epoch,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        model_fname=args.model_fname
    )
