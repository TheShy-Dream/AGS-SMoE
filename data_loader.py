import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *
from utils.convert import change_to_classify
from create_dataset import MOSI, MOSEI, PAD, UNK,UR_FUNNY,SIMS

class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "sims" in str(config.data_dir).lower():
            dataset = SIMS(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.word2id = self.word2id
        # config.pretrained_emb = self.pretrained_emb

    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    if "sims" == hp.dataset:
        bert_tokenizer = BertTokenizer.from_pretrained('/home/cjl/code/MMIM/TextBackbone/bert-base-chinese/', do_lower_case=True)
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('/home/cjl/code/MMIM/TextBackbone/bert-base-uncased/', do_lower_case=True)
    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []
        label_T = []
        label_V = []
        label_A = []
        if not hp.dataset == "sims":
            for sample in batch:
                if len(sample[0]) > 4:  # unaligned case
                    v_lens.append(torch.IntTensor([sample[0][4]]))
                    a_lens.append(torch.IntTensor([sample[0][5]]))
                else:  # aligned cases
                    v_lens.append(torch.IntTensor([len(sample[0][3])]))
                    a_lens.append(torch.IntTensor([len(sample[0][3])]))
                labels.append(torch.from_numpy(sample[1]))
                ids.append(sample[2])
        else:
            for sample in batch:
                if len(sample[0]) > 4:  # unaligned case
                    v_lens.append(torch.IntTensor([sample[0][4]]))
                    a_lens.append(torch.IntTensor([sample[0][5]]))
                else:  # aligned cases
                    v_lens.append(torch.IntTensor([len(sample[0][3])]))
                    a_lens.append(torch.IntTensor([len(sample[0][3])]))
                labels.append(torch.from_numpy(sample[1]))
                label_T.append(torch.from_numpy(sample[2]))
                label_A.append(torch.from_numpy(sample[3]))
                label_V.append(torch.from_numpy(sample[4]))
                ids.append(sample[5])


        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)

        if len(label_T)>0 and len(label_A)>0 and len(label_V)>0:
            label_T = torch.cat(label_T, dim=0)
            label_A = torch.cat(label_A, dim=0)
            label_V = torch.cat(label_V, dim=0)
            labels=labels.reshape(-1,1)
            label_T=label_T.reshape(-1,1)
            label_A=label_A.reshape(-1,1)
            label_V=label_V.reshape(-1,1)

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        v_masks = pad_sequence([torch.zeros(torch.FloatTensor(sample[0][1]).size(0)) for sample in batch], target_len=vlens.max().item(),padding_value=1)
        a_masks = pad_sequence([torch.zeros(torch.FloatTensor(sample[0][2]).size(0)) for sample in batch], target_len=vlens.max().item(),padding_value=1)

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=alens.max().item())

        ## BERT-based features input prep

        SENT_LEN = 50
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        if "token_type_ids" in bert_details[0]:
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        else:
            bert_sentence_types = torch.LongTensor([[1] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids,v_masks,a_masks,labels,label_T,label_A,label_V

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
