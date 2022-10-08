import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from datasets import load_metric

dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv', 'eval': 'val.csv'})

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def transform_labels(label):
    label = label['answer']
    return {'labels': label}

def tokenize_data(example):
    return tokenizer(example['question'], padding='max_length')

dataset = dataset.map(tokenize_data, batched=True)

remove_columns = ['video', 'frame_count', 'width', 'height', 'qid', 'type', "a0", "a1", "a2", "a3", "a4"]
dataset = dataset.map(transform_labels, remove_columns=remove_columns)


training_args = TrainingArguments("test_trainer", num_train_epochs=3, per_device_train_batch_size=16, save_steps=5000)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

train_dataset = dataset['train']
eval_dataset = dataset['eval']

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./trained_model")

eval_results = trainer.evaluate()
print(eval_results)