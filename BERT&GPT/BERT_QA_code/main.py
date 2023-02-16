
from tokenizers import BertWordPieceTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForQuestionAnswering, BertModel
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os 
import torch 
import random 
import numpy as np
import torch.backends.cudnn as cudnn
import argparse


#============================
#           ARGS
#============================
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0", help="gpus")
parser.add_argument('--max_len', type=int, default=512, help="384/512/128")
parser.add_argument('--batch_size', type=int, default=32, help="64/32")
parser.add_argument('--ckpt', type=str, default='test', help="check_point")
parser.add_argument('--epochs', type=int, default=5, help="epochs")
parser.add_argument('--lr', type=float, default=1e-5, help="lr")
args = parser.parse_args()



#============================
#           GPUS
#============================
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#============================
#      Hyperparameter
#============================
max_len = args.max_len
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
warmup_ratio = 0.3
seed = 4567
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    

#============================
#          Data
#============================
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
with open(f'./data/train_data_{max_len}.json') as f:
    train_data = json.load(f)

with open(f'./data/valid_data_{max_len}.json') as f:
    valid_data = json.load(f)

#train_data = train_data[:1000]
#valid_data = valid_data[:1000]
class MyDataset(Dataset):
    def __init__(self, tokenizer, datas):
        self.examples = []
        with tqdm(total=len(datas), desc="DATASET") as pbar:
            for idx, data in enumerate(datas):
                context = data["context"]
                question = data["question"]
                answer = data["answer"]
                token_answer = data["token_answer"]
                text_answer = data["text_answer"]
                
                tokenized_context = tokenizer.encode(context)
                tokenized_question = tokenizer.encode(question)
                context_len, question_len = len(tokenized_context.ids), len(tokenized_question.ids)
                squence_len = context_len + question_len - 1 
                padding_len = max_len - squence_len
                input_ids = tokenized_context.ids + tokenized_question.ids[1:] + [0]*padding_len
                attention_mask = [1]*squence_len + [0]*padding_len
                token_type_ids = [0]*context_len + [1]*(question_len-1) +[0]*padding_len
                assert len(token_type_ids) == len(attention_mask) == len(input_ids)
                example = {
                    "data_ids": idx,
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "token_type_ids":torch.tensor(token_type_ids),
                    "start_positions": token_answer[0],
                    "end_positions": token_answer[1]
                }
                self.examples.append(example)
                pbar.update(1)

    def __getitem__(self, idx):
        return self.examples[idx]
    def __len__(self):
        return len(self.examples)
train_dataset = MyDataset(tokenizer, train_data)
valid_dataset = MyDataset(tokenizer, valid_data)
print(f"..dataset len train:{len(train_dataset)}, valid: {len(valid_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
print(f"..dataloader len train:{len(train_dataloader)}, valid: {len(valid_dataloader)}")


#============================
#         Training
#============================

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
optimizer = AdamW(model.parameters(), lr = lr)
num_training_steps = len(train_dataloader)*epochs 
num_warmup_steps = int(num_training_steps*warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
print(f".. warmup / total steps: {num_warmup_steps}/{num_training_steps}")

model.train()
tot_train_loss = []
for e in range(1, epochs+1):
    print(f"=============[{e}/{epochs}]=============")
    train_epoch_loss = 0.0
    with tqdm(total=len(train_dataloader), desc="TRAIN") as pbar:
        for data in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids = data["input_ids"].to(device), data["attention_mask"].to(device), data["token_type_ids"].to(device)
            start_positions, end_positions = data["start_positions"].to(device), data["end_positions"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_epoch_loss += loss.item()
            tot_train_loss.append(loss.item())
            pbar.update(1)
    train_epoch_loss /= len(train_dataloader)
    print(f"..train loss: {train_epoch_loss}")



#============================
#         Validation
#============================

model.eval()
with torch.no_grad():
    valid_loss = 0.0
    accuracy = 0
    results = []
    with tqdm(total=len(valid_dataloader), desc="VALID") as vbar:
        for data in valid_dataloader:
            input_ids, attention_mask, token_type_ids = data["input_ids"].to(device), data["attention_mask"].to(device), data["token_type_ids"].to(device)
            start_positions, end_positions = data["start_positions"].to(device), data["end_positions"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
            loss = outputs['loss']
            start_token = torch.argmax(outputs['start_logits'], dim=-1)
            end_token = torch.argmax(outputs['end_logits'], dim=-1)


            if start_token == start_positions and end_token == end_positions:
                accuracy += 1 
                st, ed = start_token.item(), end_token.item()
                input_ids = input_ids[0].tolist()
                prediction = tokenizer.decode(input_ids[st:ed+1])  
                idx = data["data_ids"].item()
                result = {
                    "context": valid_data[idx]["context"],
                    "question": valid_data[idx]["question"],
                    "prediction": prediction,
                    "target": valid_data[idx]["answer"],
                }
                results.append(result)
            valid_loss += loss.item()
            vbar.update(1)
valid_loss /= len(valid_dataloader)
accuracy /= len(valid_dataloader)
print(f"..valid loss: {valid_loss}")
print(f"..valid accuracy: {accuracy}")

save_resuls = {
    "valid": {
        "loss":valid_loss, 
        "accuracy": accuracy
    
    },
    "correct":results,
    "train": tot_train_loss
}

path = f'./results/{args.ckpt}'
if not os.path.exists(path):
    os.makedirs(path)

with open(path+"/result.json", 'w') as f:
    json.dump(save_resuls, f, indent='\t')

torch.save(model.state_dict(), path+"/model.pt")












