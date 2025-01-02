from Bio import SeqIO
import sys
sys.path.append('/cluster/home/austinen/mini/ampmini')

path1="./data/AMP_1stage/AMPs_train_cdhit_40.fasta"
# path1 = "./models/data/AMP_1stage/AMPs_train_cdhit_40.fasta"
fas_id=[]
fas_seq=[]
labels=[]

for seq_record in SeqIO.parse(path1, "fasta"):
    fas_seq.append(str(seq_record.seq).upper())
    fas_id.append(str(seq_record.id))
    labels.append(1)
    break
print(len(fas_seq))

from data_feature_n import onehot_embedding

import torch
import numpy as np

def onehot_embedding_n(seq, max_len=200):

    char_list = 'ARNDCQEGHILKMFPSTWYVX'
    char_dict = {char: idx for idx, char in enumerate(char_list)}
    
    # Convert sequences to indices
    seq_indices = [[char_dict.get(char, 20) for char in each_seq] for each_seq in seq]
    
    # Convert to tensor
    seq_tensor = [torch.tensor(indices) for indices in seq_indices]
    
    # Pad sequences
    # padded_seq_tensor = torch.nn.utils.rnn.pad_sequence(seq_tensor, batch_first=True, padding_value=0)
    padded_seq_tensor = torch.stack(seq_tensor, dim=0)
    
    # Truncate sequences to max_len
    if padded_seq_tensor.size(1) > max_len:
        padded_seq_tensor = padded_seq_tensor[:, :max_len]    
    # One-hot encode
    one_hot_encoded = torch.nn.functional.one_hot(padded_seq_tensor, num_classes=21)

    if one_hot_encoded.size(1) < max_len:
        padding = torch.zeros((one_hot_encoded.size(0), max_len - one_hot_encoded.size(1),one_hot_encoded.size(2)), dtype=torch.long)
        one_hot_encoded = torch.cat((one_hot_encoded, padding), dim=1)
    
    return one_hot_encoded.float()

a = onehot_embedding_n(fas_seq)
b = onehot_embedding(fas_seq)
print(torch.all(a==b))