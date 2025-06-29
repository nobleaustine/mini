from Bio import SeqIO
import sys
import torch
import numpy as np
import time 
sys.path.append('/cluster/home/austinen/mini/ampmini')
from data_embed import onehot_embedding, BLOSUM62_embedding
from data_embed import AAI_embedding, PAAC_embedding

def compare_embeddings(old_emb, new_emb):
        return torch.allclose(old_emb, new_emb, atol=1e-6)

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

def BLOSUM62_embedding_n(seq, filepath="data/blosum62.txt", max_len=200):
    with open(filepath, "r") as f:
        text = [line.strip() for line in f if line.strip()]
    
    cha = text[0].split()
    index = np.array([list(map(float, line.split())) for line in text[1:]])
    
    BLOSUM62_dict = {char: torch.tensor(index[:, j], dtype=torch.float32) for j, char in enumerate(cha)}
    
    batch_size = len(seq)
    emb_dim = len(next(iter(BLOSUM62_dict.values())))
    embeddings = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
    
    for i, each_seq in enumerate(seq):
        seq_len = min(len(each_seq), max_len)
        embeddings[i, :seq_len] = torch.stack([BLOSUM62_dict[char] for char in each_seq[:seq_len]])
    
    return embeddings

def AAI_embedding_n(seq, filepath="data/AAindex.txt", max_len=200):
    with open(filepath, "r") as f:
        text = [line.strip() for line in f if line.strip()]
    
    cha = text[0].split('\t')[1:]
    index = np.array([list(map(float, line.split('\t')[1:])) for line in text[1:]])
    
    AAI_dict = {char: torch.tensor(index[:, j], dtype=torch.float32) for j, char in enumerate(cha)}
    AAI_dict['X'] = torch.zeros(index.shape[0], dtype=torch.float32)
    
    batch_size = len(seq)
    emb_dim = index.shape[0]
    embeddings = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
    
    for i, each_seq in enumerate(seq):
        seq_len = min(len(each_seq), max_len)
        embeddings[i, :seq_len] = torch.stack([AAI_dict.get(char, AAI_dict['X']) for char in each_seq[:seq_len]])
    
    return embeddings


if __name__ == "__main__":
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

    # a = onehot_embedding_n(fas_seq)
    # b = onehot_embedding(fas_seq)
    # print(torch.all(a==b))

    # old_embeddings = BLOSUM62_embedding(fas_seq)
    # new_embeddings = BLOSUM62_embedding_n(fas_seq)

    start_time = time.time()
    # embeddings1 = AAI_embedding_n(fas_seq)
    # embeddings1 = BLOSUM62_embedding_n(fas_seq)
    embeddings1 = onehot_embedding_n(fas_seq)
    ttime = time.time() - start_time

    print(f"Embedding1 Time: {ttime:.6f} sec | Shape: {embeddings1.shape}")

    start_time = time.time()
    # embeddings1 = AAI_embedding(fas_seq)
    # embeddings2 = BLOSUM62_embedding(fas_seq)
    embeddings2 = onehot_embedding(fas_seq)
    ttime = time.time() - start_time

    print(f"Embedding2 Time: {ttime:.6f} sec | Shape: {embeddings2.shape}")

    if compare_embeddings(embeddings1, embeddings2):
        print("Embeddings are equal")
