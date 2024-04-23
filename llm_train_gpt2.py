import datetime
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

TRAINING_DATA_PATH = "./" + ""
MODEL_NAME = "./" + "pretrained_gpt2"

# Load the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Setup the training configuration
training_args = TrainingArguments(
    output_dir="./gpt2",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=5,
    save_steps=1000,
    save_total_limit=2,
)

# Model initialization
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Training loop
training_Dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=TRAINING_DATA_PATH,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=training_Dataset,
)

# Start training
print("Training started...")
trainer.train()
print("Training completed!")

# Save the model
print("Saving the model...")
model.save_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODEL_NAME)
print("Model saved!")
