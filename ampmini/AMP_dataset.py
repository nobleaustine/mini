
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

class AMP_Dataset(Dataset):
    
    def __init__(self, csv_path="./data/AMP_sequences.csv"):
        """
        Args:
            csv_path (str): Path to the CSV file containing peptide sequences and functional labels.
        """
        self.data = pd.read_csv(csv_path, sep=",")
        # print(self.data.columns)
        self.sequences = self.data["sequence"].values
        # Extract sequence and labels as key and value pairs
        self.labels = {seq: label for seq, label in zip(
            self.data["sequence"].values, 
            self.data.iloc[:, 2:].values.astype(float))}  

    def __len__(self):
        return len(self.data)

    def BLOSUM62_embedding(self,seq,max_len=200):

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

    def onehot_embedding(self,seq,max_len=200):

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
        # print(torch.from_numpy(all_embeddings).float().shape)
        return torch.from_numpy(all_embeddings).float()

    def AAI_embedding(self,seq,max_len=200):
        
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

    def PAAC_embedding(self,seq,max_len=200):

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

        sequence = str(self.sequences[idx])

        return {
            "sequence": sequence,
            "onehot": self.onehot_embedding([sequence]),
            "blosum62": self.BLOSUM62_embedding([sequence]),
            "aai": self.AAI_embedding([sequence]),
            "paac": self.PAAC_embedding([sequence]),
            "label": torch.tensor(self.labels[sequence], dtype=torch.float32)
        }

if __name__ == "__main__":
    dataset = AMP_Dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch["sequence"])
        print(batch["aai"].shape)
        print(batch["paac"].shape)
        print(batch["blosum62"].shape)
        print(batch["onehot"].shape)
        print(batch["label"].shape)
        break



