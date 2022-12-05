import pandas as pd
import torch
from torch.utils import data
from transformers import BertTokenizer, BertModel

class TextLoader(data.Dataset):
    def __init__(
        self,
        path: str
    ):
        self.path = path
        self.csv = pd.read_csv(self.path)
        self.num_answers = 5
        self.description_key = 'result0'
        self.target_length = 50
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        example = self.csv.iloc[index]

        question = example["question"]

        answers = []

        for i in range(self.num_answers):
            answers.append(example[f'a{i}'])
        
        description = example[self.description_key]

        text_reps = []
        for answer in answers:
            text_rep = [description, question, answer]
            text_rep = " [SEP] ".join(text_rep)
            text_rep = "[CLS]" + " " + text_rep

            inputs = self.tokenizer(text_rep, return_tensors="pt")

            curr_tokens = inputs.input_ids.size()[-1]
            
            if  curr_tokens < self.target_length:
                diff = self.target_length - curr_tokens

                pad = ["[PAD]"]
                padding = " ".join(pad * diff)
                text_rep = text_rep + " " + padding
                text_reps.append(text_rep)
        
        return text_reps

if __name__ == "__main__":
    dataset = TextLoader('train_with_captions_actions.csv') 
    loader = data.DataLoader(dataset, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in loader:
        batch = map(lambda x: x[0], batch)
        batch = list(batch)

        inputs = tokenizer(batch, return_tensors="pt")
        print(inputs.input_ids.size())
        outputs = model(**inputs)
        cls_token = outputs.pooler_output
        print(cls_token.shape)
        break