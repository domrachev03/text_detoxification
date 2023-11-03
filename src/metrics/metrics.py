import gc
import tqdm
from tqdm.auto import trange
import torch
import numpy as np

from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification

import evaluate


def cleanup():
    '''Clean the RAM and VRAM from the trash'''

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_toxicity(preds, batch_size=1, device=None):
    '''Calculates toxicity of the corpus using RoBerta finetuned on toxicity classification.'''

    results = []

    model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else device
    model.to(device)

    model.eval()
    for i in tqdm.tqdm(range(0, len(preds), batch_size)):
        batch = tokenizer(preds[i:i + batch_size], return_tensors='pt', max_length=-1, padding=True).to(device)

        with torch.no_grad():
            logits = model(**batch).logits
            out = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            results.append(out)
    return 1 - np.concatenate(results)


def get_sacrebleu(inputs, preds):
    '''Calculates sacrebleu score for the inputs and predictions'''

    metric = evaluate.load("sacrebleu")

    result = metric.compute(predictions=preds, references=inputs)
    return result['score']


def get_fluency(preds, soft=False, batch_size=1, device=None):
    '''Calculates fluency of the corpus using RoBerta finetuned on CoLa dataset.'''

    model_name = 'cointegrated/roberta-large-cola-krishna2020'

    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else device
    device = device
    model.to(device)

    results = []
    for i in trange(0, len(preds), batch_size):
        batch = [t for t in preds[i: i + batch_size]]
        inputs = tokenizer(batch, max_length=-1, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out = torch.softmax(model(**inputs).logits, -1)[:, 0].cpu().numpy()
            results.append(out)
    return np.concatenate(results)


def compute_metrics(eval_preds, tokenizer=None, print_results=False, batch_size=1, device='cuda', model_name=""):
    ''' Computing metrics for the given data

    Parameters:
    eval_preds=(preds, labels) -- tuple with predictions and labels
    tokenzier -- the tokenizer for the sequence. Default: None, and sequence is treated as decoded one
    model_name -- optional model name for fancy output printing. Defaul: empty'''

    preds, labels = eval_preds

    if tokenizer is not None:
        detokenized_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        filtered_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        detokenized_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
    else:
        detokenized_preds = preds
        detokenized_labels = labels

    results = {}
    results['toxic'] = get_toxicity(detokenized_preds, batch_size=batch_size, device=device)
    results['avg_toxic'] = sum(results['toxic']) / len(results['toxic'])
    cleanup()

    results['bleu'] = get_sacrebleu(detokenized_labels, detokenized_preds) / 100
    cleanup()

    results['fluency'] = get_fluency(detokenized_preds, batch_size=batch_size, device=device)
    results['avg_fluency'] = sum(results['fluency']) / len(results['fluency'])
    cleanup()

    # count metrics
    results['joint'] = sum(results['toxic'] * results['bleu'] * results['fluency']) / len(preds)
    if print_results:
        if model_name != "":
            print("--------------")
            print(model_name)
        print("--------------")
        print("Metric   | Value")
        print("--------------")
        print(f"toxic    | {results['avg_toxic']:.2f}")
        print(f"bleu (n) | {results['bleu']:.2f}")
        print(f"fluency  | {results['avg_fluency']:.2f}")
        print("===============")
        print(f"Total    | {results['joint']:.2f}")
        print("--------------")
    return results
