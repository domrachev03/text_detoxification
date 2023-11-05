# Final solution summary
by Domrachev Ivan, B20-RO-01

## Table of content

## Dataset
The dataset with all the preprocessing could be found on [Hugging Face](https://huggingface.co/datasets/domrachev03/toxic_comments_subset). 
All the difference from the initial dataset:
1. The `Reference` and `Translation` columns are swapped so that `ref_tox > tsn_tox` 
2. For each entry, `ref_rox > 0.99` and `tsn_tox < 0.01`
   
## T5-small
The first best model is [t5-small](https://huggingface.co/t5-small) with 60m parameters. It was finetuned using the 50k train samples from the described dataset. The training parameters were the following:
``` python
max_input_length = 64
max_target_length = 64
batch_size = 128
learning_rate=2e-5
weight_decay=0.01
n_epoches = 10
```

The train history is the following:
| Epoch | Training Loss | Validation Loss | Avg Toxic | Bleu     | Fluency  | Joint    |
|-------|---------------|-----------------|-----------|----------|----------|----------|
| 1     | No log        | 1.993625        | 0.612286  | 0.224723 | 0.815205 | 0.109771 |
| 2     | 2.322900      | 1.933838        | 0.664574  | 0.231422 | 0.808446 | 0.121852 |
| 3     | 2.105600      | 1.903278        | 0.711058  | 0.234714 | 0.800925 | 0.131020 |
| 4     | 2.061800      | 1.883212        | 0.718938  | 0.238349 | 0.802042 | 0.135302 |
| 5     | 2.061800      | 1.868907        | 0.728295  | 0.239202 | 0.805758 | 0.138587 |
| 6     | 2.032600      | 1.856105        | 0.742909  | 0.241249 | 0.805957 | 0.142618 |
| 7     | 2.011700      | 1.844848        | 0.748053  | 0.244327 | 0.809981 | 0.146183 |
| 8     | 1.993500      | 1.832563        | 0.751767  | 0.246269 | 0.812085 | 0.148598 |
| 9     | 1.975400      | 1.827323        | 0.752121  | 0.246607 | 0.810365 | 0.148746 |
| 10    | 1.975400      | 1.825477        | 0.751134  | 0.246827 | 0.811284 | 0.148740 |


The score on the test dataset is:

| Metric   | Value |
|----------|------ |
| toxic    | 0.74 |
| bleu (n) | 0.24 |
| fluency  | 0.83 |
| Joint (J)    | 0.15 |

## Llama2-7b-chat
The second best model was quite tricky to train. I used a special service called [modal.com](https://modal.com/) to train & run inference in the cloud. They give around 30$ per month after registration, which is enough for the configuratino in this repository.

Another note is that the inference utilizes non-official llama2 weights from the Hugging Face, since I failed to wait upon gaining access from the Meta AI *(21 марта 2022 года российский суд признал компанию Meta экстремистской организацией и запретил ее деятельность на территории страны)*.

It was finetuned using the 25k train samples from the described dataset. The training parameters were the following:
```python 
num_epochs = 5      # Easily could be extended to 9 within the 30$ limit
batch_size = 16
lr = 3e-4
lora_config.r = 8
lora_config.alpha = 16
```

The train histori is, unfortunately, unavailable :^(
