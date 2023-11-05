from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import argparse
from collections.abc import Iterable

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore") 


def t5_wrapper(msg):
    ''' Prompt wrapper for the model '''
    wrapped = f"Make the following sentence non-toxic: '{msg}'"
    return wrapped


def t5_predict(
    requests: Iterable[str],
    batch_size: int = 1,
    model_name: str = 'domrachev03/t5_detox',
    device: str = None,
    max_length: int = 64
):
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    t5_tokenizer = AutoTokenizer.from_pretrained(model_name)

    t5_model.eval()
    t5_model.config.use_cache = False
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    msgs = [t5_wrapper(text) for text in requests]
    for i in range(0, len(msgs), batch_size):
        batch = msgs[i: i+batch_size]
        tokenized_batch = t5_tokenizer(batch, return_tensors='pt').to(device)
        with torch.no_grad():
            output = t5_model.generate(tokenized_batch['input_ids'], max_length=max_length)
            result = [t5_tokenizer.decode(out_i, skip_special_tokens=True, temperature=0) for out_i in output]
        results.extend(result)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="Text to detoxify", required=True)
    parser.add_argument('-m', "--max_length", help='Maximal length of generated output', type=int, default=64)
    parser.add_argument('-d', "--device", help='Device for computation, either "cuda" or "cpu"', type=str, default=None)

    args = parser.parse_args()

    print(t5_predict([args.input], max_length=args.max_length, device=args.device)[0])
