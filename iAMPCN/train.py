import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("/cluster/home/austinen/mini/iAMPCN")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from iAMPCN.architecture import SequenceMultiTypeMultiCNN_1, FocalLoss
from iAMPCN.data_feature import Dataset
import os
from Bio import SeqIO

def train(path="", batch_size=64, learning_rate=0.001, num_epochs=10, save_model_dir="models/AMP_1st"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fas_id=[]
    fas_seq=[]


    for seq_record in SeqIO.parse(path, "fasta"):
        fas_seq.append(str(seq_record.seq).upper())
        fas_id.append(str(seq_record.id))
    # print(len(fas_seq))
    # print(fas_id)

    train_dataset = Dataset(fasta=fas_seq)
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, max_length=200)

    model = SequenceMultiTypeMultiCNN_1(d_input=[531, 21, 23, 3], vocab_size=21, seq_len=200,
                                        dropout=0.1, d_another_h=128, k_cnn=[2, 3, 4, 5, 6], d_output=1).to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            AAI_feat = batch['seq_enc_AAI'].to(device)
            onehot_feat = batch['seq_enc_onehot'].to(device)
            BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
            PAAC_feat = batch['seq_enc_PAAC'].to(device)
            labels = batch['label'].to(device)  # Assuming the label key exists

            outputs = model(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        torch.save({'state_dict': model.state_dict()},
                   os.path.join(save_model_dir, f'textcnn_cdhit_40_{epoch}.pth.tar'))

if __name__ == "__main__":
    train(path="/cluster/home/austinen/mini/data/AMP_1stage/AMPs_train_cdhit_40.fasta")