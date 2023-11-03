from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import trange
import argparse
from collections.abc import Iterable


def wrap_messages(msgs: Iterable[str]) -> list[str]:
    ''' Wraps the message with prompt appropriate to LLAMA

    Parameters:
    msgs (Iterable[str]) -- sequence of messages to processs

    Returns:
    wrapped_msgs(list[str]) -- wrapped sequence'''

    B_INST, E_INST = "[INST] ", " [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    wrapped_msgs = [
        B_INST
        + B_SYS
        + "You are a Twitch moderator that paraphrases sentences to be non-toxic.\n"
        + E_SYS
        + "Could you paraphrase this: "
        + msg
        + "?\n"
        + E_INST
        for msg in msgs
    ]
    return wrapped_msgs


def predict(
    requests: Iterable[str],
    greb_answer: bool = False,
    batch_size: int = 1,
    max_length: int = 64
) -> Iterable[str]:
    ''' The inference function.

    Parameters:
    request (Iterable[str]): sequence of inputs into the model

    greb_answer (bool): if True, then the returned answer is only the string of the interest.
    otherwise, the raw output from the model is returned

    batch_size (int). Default: 1

    max_length (int). Default: 64
    '''


    requests = wrap_messages(requests)

    model = AutoModelForCausalLM.from_pretrained(
        'daryl149/llama-2-7b-chat-hf',
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model.load_adapter('domrachev03/llama2_7b_detoxification')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('daryl149/llama-2-7b-chat-hf')
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i in trange(0, len(requests), batch_size):
        batch = [t for t in requests[i: i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=max_length+1)
            decoded = [tokenizer.decode(out_i, skip_special_tokens=True, temperature=0) for out_i in out]

            if greb_answer:
                decoded = [
                    decoded[k][
                        decoded[k].find('[/INST]')+len('[/INST]'): decoded[k].find('</s>')
                    ]
                    for k in range(len(decoded))
                ]
            results.extend(decoded)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="Text to detoxify", required=True)
    parser.add_argument('-m', "--max_length", help='Maximal length of generated output', type=int)

    args = parser.parse_args()

    print(predict([args.input], greb_answer=True, max_length=args.max_length)[0])