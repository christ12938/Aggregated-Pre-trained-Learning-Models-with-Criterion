import random
from transformers import BertForPreTraining, BertModel, BertTokenizer, logging
import numpy as np
import torch
import time
from torch.utils.data import DataLoader


class BertPreTrainedMethod:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')


    def results(self, words):
        out = self.model(**self.tokenizer(words, add_special_tokens=True, return_tensors="pt"))
        return out.last_hidden_state.detach()


class BertFineTuneMethod:

    def __init__(self, dataset, batch_size=32, epoch=30, max_len=512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForPreTraining.from_pretrained('bert-base-uncased').to('cuda')
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.001)
        self.nspLoss = torch.nn.BCEWithLogitsLoss().to('cuda')
        self.mlmLoss = torch.nn.CrossEntropyLoss(ignore_index=0).to('cuda')
        self.batch_size = batch_size
        self.epoch = epoch
        self.print_every = 10
        self.max_len = max_len
        logging.set_verbosity_error()

    def train_one_epoch(self, epoch: int):  
        print(f"Begin epoch {epoch} ...")
        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0  
        for i, value in enumerate(self.loader):
            index = i + 1
            token_input = []
            token_target = []
            nsp_target = []
            for j in range(0, len(value[1]["sentence_1"])):
                token_input.append([value[1]["sentence_1"][j], value[1]["sentence_2"][j]])     
                token_target.append([value[1]["sentence_1"][j], value[1]["sentence_2"][j]])
                nsp_target.append([1, 0]) if value[0]["nsp"][j] == "0" else nsp_target.append([0, 1])
            token_input = self.tokenizer(token_input, add_special_tokens=True, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to('cuda')
            token_target = self.tokenizer(token_target, add_special_tokens=True, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to('cuda')

            #Mask target
            token_mask = torch.zeros(token_input['input_ids'].shape[0], token_input['input_ids'].shape[1], dtype=torch.bool).to('cuda')
            for x in range(0, token_input['input_ids'].shape[0]):
                for y in range(0, token_input['input_ids'].shape[1]):
                    if random.random() < 0.15:
                        if random.random() < 0.1:
                            token_input['input_ids'][x, y] = random.randint(1000, self.tokenizer.vocab_size - 1)
                            token_mask[x, y] = True
                        else:
                            token_input['input_ids'][x, y] = 103
                            token_mask[x, y] = True
                    else:
                        token_mask[x, y] = False
            #Add NSP
            nsp_target = torch.FloatTensor(nsp_target).to('cuda')

            self.optimizer.zero_grad()  
            outputs = self.model(**token_input)
            prediction_logits = outputs.prediction_logits
            seq_relationship_logits = outputs.seq_relationship_logits

            #Mask output
            token_target['input_ids'][token_mask != True] = 0

            #Loss input
            loss_nsp = self.nspLoss(seq_relationship_logits.to('cuda'), nsp_target)
            loss_token = self.mlmLoss(prediction_logits.transpose(1, 2).to('cuda'), token_target['input_ids'])  
    
            loss = loss_token + loss_nsp  
            average_nsp_loss += loss_nsp  
            average_mlm_loss += loss_token
    
            loss.backward()  
            self.optimizer.step()  

            #For logging
            if index % self.print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                self.write_summary(epoch, elapsed, index, average_nsp_loss, average_mlm_loss, prediction_logits, seq_relationship_logits, token_target['input_ids'], nsp_target, token_mask)
                average_nsp_loss = 0
                average_mlm_loss = 0
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, f"checkpoints/checkpoint_{epoch}")
        return loss

    def write_summary(self, epoch, elapsed, index, average_nsp_loss, average_mlm_loss, token, nsp, token_target, nsp_target, token_mask):
        nsp_acc = self.nsp_accuracy(nsp, nsp_target)
        token_acc = self.token_accuracy(token, token_target, token_mask)
        print(f"Epoch: [ {epoch} / {self.epoch} ]")
        print(f"Batch: [ {index} / {len(self.loader)} ]")
        print(f"Avg NSP Loss: {average_nsp_loss}, Avg MLM Loss: {average_mlm_loss}")
        print(f"NSP Accuracy: {nsp_acc}, Token Accuracy: {token_acc}")
        print(f"Time elapsed: {elapsed.tm_hour}:{elapsed.tm_min}:{elapsed.tm_sec}")



    def nsp_accuracy(self, result: torch.Tensor, target: torch.Tensor):
        s = (result.argmax(1) == target.argmax(1)).sum()  
        return round(float(s / result.size(0)), 2)

    def token_accuracy(self, result: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor):
        r = result.argmax(-1).masked_select(token_mask)
        t = target.masked_select(token_mask)  
        s = (r == t).sum()  
        return round(float(s / (result.size(0) * result.size(1))), 2)

