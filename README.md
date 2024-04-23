# llm_training_playground
Playground for experiementing with pre-trained LLMs using HuggingFace Transformers.

## Datasets
Datasets for LLM training can be found from various sources. As an example, you can search for training data on [Kaggle](https://www.kaggle.com/datasets/chaitanyakck/medical-text).

## Setup (GPT2 example)
1. Create a new venv:
```sh
$ python3 -m venv [<venv name>]
$ source [<venv name>]/bin/activate
$ pip3 install -r requirements.txt  # This takes a while
```
2. Download and place your dataset in `/data`, adjust `TRAINING_DATA_PATH` in `llm_train_gpt2.py`.
3. Run the training script via:
```sh
$ python3 llm_train_gpt2.py
```
This'll take a while, like 45-60 minutes, depending on how beefy your PC is. The model will be saved in `./pretrained_gpt2`.

### Testing the model
1. Write your prompt in the `PROMPT` variable in `llm_gpt2.py`
2. Run the Python script via:
```sh
$ python3 llm_gpt2.py
```
This'll take a few minutes.
4. Adjust your model, re-train, try to get better results, and have fun B-)
