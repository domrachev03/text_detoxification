from transformers import AutoModelForCausalLM, AutoTokenizer
from collections.abc import Iterable
import torch
import numpy as np

def wrap_messages(msgs):
    B_INST, E_INST = "[INST] ", " [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    prefixed_queries = [
        B_INST
        + B_SYS
        + "You are a Twitch moderator that paraphrases sentences to be non-toxic.\n"
        + E_SYS
        + "Could you paraphrase this: "
        + query
        + "?\n"
        + E_INST
        for query in queries
    ]
    return prefixed_queries


def predict(requests, grab_answer=False, batch_size=1, max_length=64):
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
    for i in range(0, len(requests), batch_size):
        print(f"Iteration {i}")
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

            if grab_answer:
                decoded = [
                    decoded[i][len(batch[i]):decoded[i].find('</s>')] 
                    for i in range(batch_size)
                ]

            results.extend(decoded)

    return results


if __name__=='__main__':
    queries = [
        'Fuck you!', 
        'This freaking chair makes me nuts', 
        'This fucking sause, I love it', 
        'I hate gays'
    ]

    print(predict(queries, grab_answer=True, batch_size=2))
