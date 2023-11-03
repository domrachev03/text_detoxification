import datasets
from llama_recipes.datasets.utils import Concatenator


def map_fn(examples: datasets.Dataset, tokenizer):
    B_INST, E_INST = "[INST] ", " [/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    inputs = [
        B_INST
        + B_SYS
        + "You are a Twitch moderator that paraphrases sentences to be non-toxic.\n"
        + E_SYS
        + "Could you paraphrase this: "
        + ref
        + "?\n"
        + E_INST
        + tsn
        + "</s>"
        for ref, tsn in zip(examples["reference"], examples["translation"])
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        return_overflowing_tokens=False
    )

    return model_inputs


def get_custom_dataset(
    dataset_config,
    tokenizer,
    split,
    n_samples=(25000, 1000),
    batch_size=64
):
    dataset = datasets.load_dataset("domrachev03/toxic_comments_subset")
    dataset['train'] = dataset['train'].select(range(n_samples[0]))
    dataset['test'] = dataset['test'].select(range(n_samples[1]))

    dataset = dataset.map(
        lambda x: map_fn(x, tokenizer=tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
    )["train" if split == dataset_config.train_split else "test"]   

    dataset = dataset.map(Concatenator(), batched=True, batch_size=None)
    return dataset
