import torch
from torch.utils.data import Dataset, DataLoader
import json
from cs336_alignment import utils
import random
from tqdm import tqdm

class SFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.tokenized_data = []

        for i, example in enumerate(tqdm(utils.read_jsonl_stream(dataset_path, compressed=True))):
            instruction = example['prompt']
            response = example['response']
            prompt_format = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"""
            prompt_format = tokenizer.bos_token + prompt_format + tokenizer.eos_token
            if i < 5:
                print(prompt_format)
            tokenized_example = self.tokenizer(prompt_format, add_special_tokens=False, padding=False, return_tensors="pt").input_ids[0]
            #print(tokenized_example)
            self.tokenized_data.append(tokenized_example)
        
        
        # concatenate list of tensors into one tensor
        self.tokenized_data = torch.cat(self.tokenized_data, dim=0)

        # get the divisor of the length of the tensor
        
        self.num_sequences = (len(self.tokenized_data) - 1) // self.seq_length

        # reshape this tensor into a 2D tensor with shape (num_sequences, seq_length) while dropping the last one if it is not equal to seq_length
        
        self.sequenced_data_input = self.tokenized_data[:self.num_sequences*self.seq_length].view(-1, self.seq_length)
        self.sequenced_data_target = self.tokenized_data[1:self.num_sequences*self.seq_length+1].view(-1, self.seq_length)

        if shuffle:
            idx = torch.randperm(self.sequenced_data_input.nelement())
            self.sequenced_data_input = self.sequenced_data_input.view(-1)[idx].view(self.sequenced_data_input.size())
            self.sequenced_data_target = self.sequenced_data_target.view(-1)[idx].view(self.sequenced_data_target.size())
        self.length = len(self.sequenced_data_input)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.sequenced_data_input[i], "labels": self.sequenced_data_target[i]}

def get_batches_sft(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

