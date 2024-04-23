# llm_training_playground
Playground for experiementing with pre-trained LLMs using HuggingFace Transformers.

## Datasets
Datasets for LLM training can be found from various sources. As an example, you can search for training data on [Kaggle](https://www.kaggle.com/datasets/chaitanyakck/medical-text).

## Setup
1. Create a new venv:
```sh
$ python3 -m venv [<venv name>]
$ source [<venv name>]/bin/activate
$ pip3 install -r requirements.txt  # This takes a while
```
2. Download and place your dataset in `/data`, adjust `TRAINING_DATA_PATH` in `llm_train_study.py`.
