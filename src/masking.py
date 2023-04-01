import random
import torch


# TODO: Change Rates
class BertMasker:
    def __init__(self, tokenizer, mask_rate_normal=0.15, mask_rate_special=0.85, replace_rate=0, unchanged_rate=0.01):
        self.tokenizer = tokenizer
        # self.original_vocab = original_vocab
        # self.new_vocab = new_vocab
        self.mask_rate_normal = mask_rate_normal
        self.mask_rate_special = mask_rate_special
        self.replace_rate = replace_rate
        self.unchanged_rate = unchanged_rate

    # TODO: Implement switching words, Check Input ID Reference
    def mask(self, input_ids):
        subword_mask = torch.zeros(input_ids.shape).to('cuda')
        maskword_mask = torch.zeros(input_ids.shape).to('cuda')
        for sen_idx in range(input_ids.shape[0]):
            for wrd_idx in range(input_ids.shape[1]):
                if input_ids[sen_idx, wrd_idx] == self.tokenizer.pad_token_id:
                    # Break if id is PAD token
                    break
                elif self.tokenizer.decode([input_ids[sen_idx, wrd_idx]]).startswith("##"):
                    subword_mask[sen_idx, wrd_idx] = 1
                    if input_ids[sen_idx, wrd_idx - 1] == self.tokenizer.mask_token_id:
                        input_ids[sen_idx, wrd_idx] = self.tokenizer.mask_token_id
                        maskword_mask[sen_idx, wrd_idx] = 1
                elif input_ids[sen_idx, wrd_idx] != self.tokenizer.cls_token_id \
                        and input_ids[sen_idx, wrd_idx] != self.tokenizer.sep_token_id:
                    # Mask words
                    if random.random() < self.mask_rate_normal:
                        if random.random() < 1 - self.unchanged_rate - self.replace_rate:
                            #Mask Word
                            input_ids[sen_idx, wrd_idx] = self.tokenizer.mask_token_id
                            maskword_mask[sen_idx, wrd_idx] = 1
        return subword_mask, maskword_mask
