import sys
sys.path.append("./")
sys.path.append("../")

import os
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.utils.data
import vocab


def BLOSUM62_embedding(seq,max_len=200):

    f=open("data/blosum62.txt") # "iAMPCN/data/blosum62.txt"
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split(' ')
    while '' in cha:
        cha.remove('')
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    BLOSUM62_dict={}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]]=index[:,j]
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(BLOSUM62_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),23))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def onehot_embedding(seq,max_len=200):

    char_list='ARNDCQEGHILKMFPSTWYVX'
    char_dict={}
    for i in range(len(char_list)):
        char_dict[char_list[i]]=i
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            codings=np.zeros(21)
            if each_char in char_dict.keys():
                codings[char_dict[each_char]]=1
            else:
                codings[20]=1
            temp_embeddings.append(codings)
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),21))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
              
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def AAI_embedding(seq,max_len=200):
    
    f=open('data/AAindex.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha=cha[1:]
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp=temp[1:]
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    AAI_dict={}
    for j in range(len(cha)):
        AAI_dict[cha[j]]=index[:,j]
    AAI_dict['X']=np.zeros(531)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),531))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def PAAC_embedding(seq,max_len=200):

    f=open('data/PAAC.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha=cha[1:]
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp=temp[1:]
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    AAI_dict={}
    for j in range(len(cha)):
        AAI_dict[cha[j]]=index[:,j]
    AAI_dict['X']=np.zeros(3)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),3))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def PC6_embedding(seq,max_len=200):
    f=open('data/6-pc')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    text=text[1:]
    AAI_dict={}
    for each_line in text:
        temp=each_line.split(' ')
        while '' in temp:
            temp.remove('')
        for i in range(1,len(temp)):
            temp[i]=float(temp[i])
        AAI_dict[temp[0]]=temp[1:]
    AAI_dict['X']=np.zeros(6)
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len>len(each_seq):
            zero_padding=np.zeros((max_len-len(each_seq),6))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(each_seq):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()

def index_encoding(sequences,max_len=200):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130

    Parameters
    ----------
    sequences: list of equal-length sequences

    Returns
    -------
    np.array with shape (#sequences, length of sequences)
    '''
    seq_list=[]
    for s in sequences:
        temp=list(s)
        while len(temp)<max_len:
            temp.append(20)
        temp=temp[:max_len]
        seq_list.append(temp)

    df = pd.DataFrame(seq_list)
    encoding = df.replace(vocab.AMINO_ACID_INDEX)
    encoding = encoding.values.astype(int)
    return encoding

class DataEmbeder:
    def __init__(self, fasta_path ="./data/AMP_s1/train" , save_dir="data/AMP_s1/train/embeded_data", max_length=200):

        self.fasta_path = fasta_path
        self.save_dir = save_dir
        self.max_length = max_length

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.peptide_data = self._read_fasta()
        self._embed_and_save()  

    def _read_fasta(self):

        """Reads sequences from a FASTA file and extracts peptide IDs and sequences."""
        path1 = os.path.join(self.fasta_path, "train_AMPs_cdhit_40.fasta")
        path2 = os.path.join(self.fasta_path, "train_nonAMPs_cdhit_40.fasta")
        peptide_data = []
        print("Loading data for embedding ...")
        label = 1
        # id, seq, label
        # loading amp data
        for seq_record in SeqIO.parse(path1, "fasta"):
            peptide_data.append((str(seq_record.id), str(seq_record.seq).upper(),label))
        # print(len(peptide_data))

        label = 0
        # loading non-amp data
        for seq_record in SeqIO.parse(path2, "fasta"):
            peptide_data.append((str(seq_record.id), str(seq_record.seq).upper(),label))
        # print(len(peptide_data))
        print("Data loaded successfully!")

        return peptide_data

    def _embed_and_save(self):


        """Embed sequences and saves each embedding separately by peptide ID."""

        print("Embedding started...")
        for peptide_id, sequence, label in self.peptide_data:
            peptide_dir = os.path.join(self.save_dir, peptide_id)

            if not os.path.isdir(peptide_dir):
                os.makedirs(peptide_dir, exist_ok=True)
                print(f"Embedding for {peptide_id}...")
                np.save(os.path.join(peptide_dir, "onehot.npy"), onehot_embedding([sequence]))
                np.save(os.path.join(peptide_dir, "blosum62.npy"), BLOSUM62_embedding([sequence]))
                np.save(os.path.join(peptide_dir, "aai.npy"), AAI_embedding([sequence]))
                np.save(os.path.join(peptide_dir, "paac.npy"), PAAC_embedding([sequence]))
                np.save(os.path.join(peptide_dir, "pc6.npy"), PC6_embedding([sequence]))
                np.save(os.path.join(peptide_dir, "index.npy"), index_encoding([sequence]))
                np.save(os.path.join(peptide_dir, "label.npy"), np.array([label]))
            else:
                print(f"Embedding for {peptide_id} already exists. Skipping...")

        print("All embedding saved successfully!")

class AMP_Dataset(Dataset):

    def __init__(self, fasta_path="./data/AMP_s1/train", save_dir="./data/AMP_s1/train/embeded_data"):

        self.save_dir = save_dir
        self.peptide_data = self._read_fasta(fasta_path)

    def _read_fasta(self, fasta_path):
        """Reads FASTA file and extracts peptide IDs and sequences."""
        path1 = os.path.join(fasta_path, "train_AMPs_cdhit_40.fasta")
        path2 = os.path.join(fasta_path, "train_nonAMPs_cdhit_40.fasta")
        l1 = [(record.id, str(record.seq).upper()) for record in SeqIO.parse(path1, "fasta")]
        l2 = [(record.id, str(record.seq).upper()) for record in SeqIO.parse(path2, "fasta")]
        return l1 + l2

    def __len__(self):
        return len(self.peptide_data)

    def __getitem__(self, idx):
        peptide_id, sequence = self.peptide_data[idx]
        peptide_dir = os.path.join(self.save_dir, peptide_id)

        return {
            "peptide_id": peptide_id,
            "sequence": sequence,
            "seq_enc": torch.tensor(np.load(os.path.join(peptide_dir, "index.npy")), dtype=torch.float32),
            "seq_enc_onehot": torch.tensor(np.load(os.path.join(peptide_dir, "onehot.npy")), dtype=torch.float32),
            "seq_enc_BLOSUM62": torch.tensor(np.load(os.path.join(peptide_dir, "blosum62.npy")), dtype=torch.float32),
            "seq_enc_PAAC": torch.tensor(np.load(os.path.join(peptide_dir, "paac.npy")), dtype=torch.float32),
            "seq_enc_AAI": torch.tensor(np.load(os.path.join(peptide_dir, "aai.npy")), dtype=torch.float32),
            "seq_enc_PC6": torch.tensor(np.load(os.path.join(peptide_dir, "pc6.npy")), dtype=torch.float32),
            "label": torch.tensor(np.load(os.path.join(peptide_dir, "label.npy")), dtype=torch.float32)
        }

if __name__ == "__main__":
    DataEmbeder()
    dataset = AMP_Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch["sequence"])
        print(batch["seq_enc"].shape)
        print(batch["seq_enc_onehot"].shape)
        print(batch["seq_enc_BLOSUM62"].shape)
        print(batch["seq_enc_PAAC"].shape)
        print(batch["seq_enc_AAI"].shape)
        print(batch["seq_enc_PC6"].shape)
        print(batch["label"])
        break


class AMP_Dataset(Dataset):
    
    def __init__(self, csv_path="./data/AMP_sequences.csv"):
        """
        Args:
            csv_path (str): Path to the CSV file containing peptide sequences and functional labels.
        """
        self.data = pd.read_csv(csv_path, sep="\t")

        # Extract sequence and labels as key and value pairs
        self.data = {seq: label for seq, label in zip(
            self.data["sequence"].values, 
            self.data.iloc[:, 2:].values.astype(float))}  

    def __len__(self):
        return len(self.data)

    def BLOSUM62_embedding(seq,max_len=200):

        f=open("data/blosum62.txt") 
        text=f.read()
        f.close()
        text=text.split('\n')
        while '' in text:
            text.remove('')
        cha=text[0].split(' ')
        while '' in cha:
            cha.remove('')
        index=[]
        for i in range(1,len(text)):
            temp=text[i].split(' ')
            while '' in temp:
                temp.remove('')
            for j in range(len(temp)):
                temp[j]=float(temp[j])
            index.append(temp)
        index=np.array(index)
        BLOSUM62_dict={}
        for j in range(len(cha)):
            BLOSUM62_dict[cha[j]]=index[:,j]
        all_embeddings=[]
        for each_seq in seq:
            temp_embeddings=[]
            for each_char in each_seq:
                temp_embeddings.append(BLOSUM62_dict[each_char])
            if max_len>len(each_seq):
                zero_padding=np.zeros((max_len-len(each_seq),23))
                data_pad=np.vstack((temp_embeddings,zero_padding))
            elif max_len==len(each_seq):
                data_pad=temp_embeddings
            else:
                data_pad=temp_embeddings[:max_len]
            all_embeddings.append(data_pad)
        all_embeddings=np.array(all_embeddings)
        return torch.from_numpy(all_embeddings).float()

    def onehot_embedding(seq,max_len=200):

        char_list='ARNDCQEGHILKMFPSTWYVX'
        char_dict={}
        for i in range(len(char_list)):
            char_dict[char_list[i]]=i
        all_embeddings=[]
        for each_seq in seq:
            temp_embeddings=[]
            for each_char in each_seq:
                codings=np.zeros(21)
                if each_char in char_dict.keys():
                    codings[char_dict[each_char]]=1
                else:
                    codings[20]=1
                temp_embeddings.append(codings)
            if max_len>len(each_seq):
                zero_padding=np.zeros((max_len-len(each_seq),21))
                data_pad=np.vstack((temp_embeddings,zero_padding))
            elif max_len==len(each_seq):
                data_pad=temp_embeddings
            else:
                data_pad=temp_embeddings[:max_len]
                
            all_embeddings.append(data_pad)
        all_embeddings=np.array(all_embeddings)
        return torch.from_numpy(all_embeddings).float()

    def AAI_embedding(seq,max_len=200):
        
        f=open('data/AAindex.txt')
        text=f.read()
        f.close()
        text=text.split('\n')
        while '' in text:
            text.remove('')
        cha=text[0].split('\t')
        while '' in cha:
            cha.remove('')
        cha=cha[1:]
        index=[]
        for i in range(1,len(text)):
            temp=text[i].split('\t')
            while '' in temp:
                temp.remove('')
            temp=temp[1:]
            for j in range(len(temp)):
                temp[j]=float(temp[j])
            index.append(temp)
        index=np.array(index)
        AAI_dict={}
        for j in range(len(cha)):
            AAI_dict[cha[j]]=index[:,j]
        AAI_dict['X']=np.zeros(531)
        all_embeddings=[]
        for each_seq in seq:
            temp_embeddings=[]
            for each_char in each_seq:
                temp_embeddings.append(AAI_dict[each_char])
            if max_len>len(each_seq):
                zero_padding=np.zeros((max_len-len(each_seq),531))
                data_pad=np.vstack((temp_embeddings,zero_padding))
            elif max_len==len(each_seq):
                data_pad=temp_embeddings
            else:
                data_pad=temp_embeddings[:max_len]
            all_embeddings.append(data_pad)
        all_embeddings=np.array(all_embeddings)
        return torch.from_numpy(all_embeddings).float()

    def PAAC_embedding(seq,max_len=200):

        f=open('data/PAAC.txt')
        text=f.read()
        f.close()
        text=text.split('\n')
        while '' in text:
            text.remove('')
        cha=text[0].split('\t')
        while '' in cha:
            cha.remove('')
        cha=cha[1:]
        index=[]
        for i in range(1,len(text)):
            temp=text[i].split('\t')
            while '' in temp:
                temp.remove('')
            temp=temp[1:]
            for j in range(len(temp)):
                temp[j]=float(temp[j])
            index.append(temp)
        index=np.array(index)
        AAI_dict={}
        for j in range(len(cha)):
            AAI_dict[cha[j]]=index[:,j]
        AAI_dict['X']=np.zeros(3)
        all_embeddings=[]
        for each_seq in seq:
            temp_embeddings=[]
            for each_char in each_seq:
                temp_embeddings.append(AAI_dict[each_char])
            if max_len>len(each_seq):
                zero_padding=np.zeros((max_len-len(each_seq),3))
                data_pad=np.vstack((temp_embeddings,zero_padding))
            elif max_len==len(each_seq):
                data_pad=temp_embeddings
            else:
                data_pad=temp_embeddings[:max_len]
            all_embeddings.append(data_pad)
        all_embeddings=np.array(all_embeddings)
        return torch.from_numpy(all_embeddings).float()


    def __getitem__(self, idx):
        sequence = list(self.data_dict.keys())[idx]
        sequence = self.sequences[idx]

        return {
            "onehot": onehot_embedding([sequence]),
            "sequence": sequence,
            "seq_enc": self._embed_sequence(sequence),  # Dummy embedding
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

if __name__ == "__main__":
    dataset = AMP_Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch["sequence"])
        print(batch["seq_enc"].shape)  # Should be (batch_size, sequence_length, embedding_dim)
        print(batch["label"].shape)  # Should be (batch_size, num_labels)
        break
