
import sys
sys.path.append("./")

import os
import logging
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from architecture import SequenceTransformer, FocalLoss
from AMP_dataset import AMP_Dataset

from torch.utils.tensorboard import SummaryWriter


# Setup logging
log_dir = "logs/SequenceTransformer/AMP_s2"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def train(path="", batch_size=64, learning_rate=0.001, num_epochs=100, save_model_dir="n_models/SequenceTransformer/AMP_s2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")
    torch.autograd.set_detect_anomaly(True)

    train_dataset = AMP_Dataset(csv_path=path)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logging.info("Dataset and Dataloader created...")

    model = SequenceTransformer().to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            # print(batch.shape)
            AAI_embed = batch["aai"].squeeze(1).to(device)
            PAAC_embed = batch["paac"].squeeze(1).to(device)
            BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
            OH_embed = batch["onehot"].squeeze(1).to(device)
            labels = batch["label"].squeeze(1).to(device)
            # print("AAI_embed", AAI_embed.shape)
            # print("PAAC_embed", PAAC_embed.shape)
            # print("BLOSUM62_embed", BLOSUM62_embed.shape)
            # print("OH_embed", OH_embed.shape)
            # print("labels", labels.shape)
            x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]
            # labels = labels.unsqueeze(1).float()

            outputs = model(x)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                msg = f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                logging.info(msg)
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar('epoch loss', epoch_loss, epoch)

        os.makedirs(save_model_dir, exist_ok=True)
        if epoch % 100 == 0:
            model_path = os.path.join(save_model_dir, f'ST{epoch}.pth.tar')
            torch.save({'state_dict': model.state_dict()}, model_path)
            logging.info(f"Model saved at {model_path}")

if __name__ == "__main__":
    train(path="./data/AMP_sequences.csv")

