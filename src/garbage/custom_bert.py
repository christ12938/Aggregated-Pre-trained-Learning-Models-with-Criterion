from transformers import BertForPreTraining, BertTokenizer, logging, BertConfig
import torch
from torch.utils.data import DataLoader
from masking import BertMasker


class CustomBert:

    def __init__(self, dataset, batch_size=8, max_len=512, vocab_size=30552):
        self.backbone = BertForPreTraining.from_pretrained('bert-base-uncased').to('cuda')
        self.ext_bert = BertForPreTraining(BertConfig(vocab_size=vocab_size)).to('cuda')
        self.backbone_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_masker = BertMasker(tokenizer=self.backbone_tokenizer)
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.ext_bert.parameters(), lr=0.00001, weight_decay=0.001)
        self.nspLoss = torch.nn.BCEWithLogitsLoss().to('cuda')
        self.mlmLoss = torch.nn.CrossEntropyLoss(ignore_index=0).to('cuda')
        self.max_len = max_len
        self.vocab_size = vocab_size
        logging.set_verbosity_error()

    # TODO: Change to_cuda
    def train_one_epoch(self):
        for i, value in enumerate(self.loader):
            print(f"Batch ... [ {str(i + 1)} / {str(len(self.loader))} ]")
            # Inference
            token_input = []
            nsp_target = []
            for j in range(len(value["sentence_1"])):
                token_input.append([value["sentence_1"][j], value["sentence_2"][j]])
                nsp_target.append([1, 0]) if value["nsp"][j] == 0 else nsp_target.append([0, 1])
            token_input = self.backbone_tokenizer(token_input, add_special_tokens=True, padding=True, truncation=True,
                                                  max_length=self.max_len, return_tensors="pt").to('cuda')
            subword_mask, maskword_mask = self.bert_masker.mask(input_ids=token_input["input_ids"].to('cuda'))
            # Get the last hidden state embeddings
            backbone_out = self.backbone(**token_input, output_hidden_states=True).hidden_states[-1]
            ext_inputs, ext_in_maskword_mask = self.mean_subwords(backbone_embeddings=backbone_out,
                                                                  subword_mask=subword_mask,
                                                                  maskword_mask=maskword_mask,
                                                                  attention_mask=token_input['attention_mask'])
            ext_out = self.ext_bert(**ext_inputs)

            self.optimizer.zero_grad()
            print(ext_out.prediction_logits.shape)
            # Mask output
            # TODO: Change token target implementation
            token_target = torch.zeros()
            token_target['input_ids'][ext_in_maskword_mask != True] = 0

            # Loss input
            loss_nsp = self.nspLoss(ext_out.seq_relationship_logits, nsp_target)
            loss_token = self.mlmLoss(ext_out.prediction_logits.transpose(1, 2), token_target['input_ids'])

            loss = loss_token + loss_nsp
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_token

            loss.backward()
            self.optimizer.step()

    @staticmethod
    def mean_subwords(backbone_embeddings, subword_mask, maskword_mask, attention_mask):
        ext_in_embeddings = torch.zeros(backbone_embeddings.shape[0],
                                        backbone_embeddings.shape[1] - int(torch.min(torch.sum(subword_mask, dim=1)).item()),
                                        backbone_embeddings.shape[2]).to('cuda')
        ext_in_maskword_mask = torch.zeros(ext_in_embeddings.shape[0], ext_in_embeddings.shape[1]).to('cuda')
        ext_in_attention_mask = torch.zeros(ext_in_embeddings.shape[0], ext_in_embeddings.shape[1]).to('cuda')
        for sen_idx in range(backbone_embeddings.shape[0]):
            mean_embeddings = mean_count = idx_offset = 0
            for wrd_idx in range(backbone_embeddings.shape[1]):
                if subword_mask[sen_idx, wrd_idx]:
                    mean_embeddings += backbone_embeddings[sen_idx, wrd_idx]
                    mean_count += 1
                    idx_offset += 1
                    if not subword_mask[sen_idx, wrd_idx - 1]:
                        mean_embeddings += backbone_embeddings[sen_idx, wrd_idx - 1]
                        mean_count += 1
                    if wrd_idx == backbone_embeddings.shape[1] - 1 or not subword_mask[sen_idx, wrd_idx + 1]:
                        ext_in_embeddings[sen_idx, wrd_idx - idx_offset] = mean_embeddings / mean_count
                        mean_embeddings = 0
                        mean_count = 0
                if maskword_mask[sen_idx, wrd_idx] and (wrd_idx == 0 or not maskword_mask[sen_idx, wrd_idx - 1]):
                    ext_in_maskword_mask[sen_idx, wrd_idx - idx_offset] = 1
            ext_in_attention_mask[sen_idx][:int((torch.sum(attention_mask[sen_idx]) - torch.sum(subword_mask[sen_idx])).item())] = 1
        return {'inputs_embeds': ext_in_embeddings,
                'attention_mask': ext_in_attention_mask}, ext_in_maskword_mask
