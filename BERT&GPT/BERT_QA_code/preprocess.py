
import os 
import pandas as pd 
import numpy  as np 
import json 
import sys 
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm 

train_file = './data/train.json'
validation_file = './data/dev.json'

with open(train_file) as f:
    raw_train_data = json.load(f)

with open(validation_file) as f:
    raw_valid_data = json.load(f)
    

# Load pre-trained model tokenizer (vocabulary)
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
max_len = 128 #384 #512

class SquadSample:
    def __init__(self, context, question, basic_answer, more_answer, start_idx):
        self.context = context
        self.question = question
        self.basic_answer = basic_answer
        self.more_answer = more_answer
        self.start_idx = start_idx
        self.end_idx = None 
        self.start_idx_token = start_idx
        self.end_idx_token = None
        self.offsets = None 
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.validExample = None


    def preprocess(self):

        self.context = " ".join(str(self.context).split())
        self.question = " ".join(str(self.question).split())
        contextTokenizer = tokenizer.encode(self.context)

        if self.basic_answer is not None: # answer O
            self.basic_answer = " ".join(str(self.basic_answer).split())


            self.end_idx = self.start_idx + len(self.basic_answer)                  
            if self.end_idx >= len(self.context): 
                self.validExample = False #When the context length and the answer length do not match
                return
            
            is_part_of_answer = [0]*len(self.context)
            for i in range(self.start_idx, self.end_idx):
                is_part_of_answer[i] = 1 
            
            answer_id_token = []
            for idx, (start, end) in enumerate(contextTokenizer.offsets):
                if (sum(is_part_of_answer[start:end]) > 0):
                    answer_id_token.append(idx)
            if len(answer_id_token) == 0: 
                self.validExample = False # No answers 
                return
            
            self.start_idx_token = answer_id_token[0]
            self.end_idx_token = answer_id_token[-1]

        self.offsets = contextTokenizer.offsets
        questionTokinizer = tokenizer.encode(self.question)
        self.input_ids = contextTokenizer.ids + questionTokinizer.ids[1:]
        padding_length = max_len - len(self.input_ids)            
        if padding_length < 0:
            self.validExample= False  # Exceeds max_len 
            return 
    
        result = {
            "context":self.context,
            "question":self.question, 
            "answer":self.basic_answer,
            "token_answer":[self.start_idx_token, self.end_idx_token], 
            "text_answer":[self.start_idx, self.end_idx]
        }
        return result



def create_squad_examples(raw_data):
    length = len(raw_data["data"])
    print(f"contexts len {length}")
    results = []
    cnt = 0
    with tqdm(total=length, desc ='datas') as pbar:
        for item in raw_data["data"]:
            for para in item["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    question = qa["question"]
                    cnt += 1
                    if qa["answers"]:
                        answer_text = qa["answers"][0]["text"] #first answer
                        all_answers = [_["text"] for _ in qa["answers"]] #all answer 
                        start_char_idx = qa["answers"][0]["answer_start"] #first answer idx
                        squad_eg = SquadSample(context, question, answer_text, all_answers, start_char_idx)
                        result = squad_eg.preprocess()
                        if result == None: continue
                        results.append(result)
            pbar.update(1)
    print(f"BF LEN {cnt}")
    print(f"AF LEN {len(results)}")
    return results


train_data = create_squad_examples(raw_train_data) 
valid_data = create_squad_examples(raw_valid_data) 

 
with open(f"./data/train_data_{max_len}.json", 'w') as f:
    json.dump(train_data, f, indent='\t')
    

with open(f"./data/valid_data_{max_len}.json", 'w') as f:
    json.dump(valid_data, f, indent='\t')
    


