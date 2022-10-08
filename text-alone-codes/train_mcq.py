from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import numpy as np


next = load_dataset("next-dataset")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

answers = ["a0", "a1", "a2", "a3", "a4"]
def preprocess_function(examples):
    first_sentences = [[context] * 5 for context in examples["question"]]
    question_headers = [""] * len(examples["question"])
    second_sentences = [
        [f"{examples[end][i]}" for end in answers] for i, header in enumerate(question_headers)
    ]
    # return {"": [""] for i in range(len(examples))}

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


tokenized_next = next.map(preprocess_function, batched=True)


def preprocess_label(examples):
    return {"label": [int(example) for example in examples["answer"]]}

tokenized_next = tokenized_next.map(preprocess_label, batched=True)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" # if "answer" in features[0].keys() else "answers"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch



model = AutoModelForMultipleChoice.from_pretrained("./trained_model")

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(labels, logits)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=5000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_next["train"],
    eval_dataset=tokenized_next["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# trainer.train()

# model.load_pretrained("./trained_model")

# model.save_pretrained("./trained_model")

eval_results = trainer.evaluate()
print(eval_results)