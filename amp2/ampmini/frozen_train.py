from transformers import AutoTokenizer, EsmForProteinFolding
import sys
sys.path.append("./")

import os
import time
import logging
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from network import FocalLoss, TranCNN
from AMP_Dataset import AMP_Dataset

# Setup logging
log_dir = "logs/s1AMP/TranCNN/"
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

def train(path="./data/s1_train_AMPseqs.csv", batch_size=64, learning_rate=0.001, num_epochs=102, save_model_dir="models/s1AMP/TranCNN/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    # Load ESM model and tokenizer
    esm_model = EsmForProteinFolding.from_pretrained("../esmfold_v1_local", local_files_only=True).to(device)
    esm_tokenizer = AutoTokenizer.from_pretrained("../esmfold_v1_local", local_files_only=True)
    esm_model.eval()

    if torch.cuda.device_count() > 1:
        esm_model = DataParallel(esm_model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for ESM model.")
    else:
        logging.info("Using single GPU for ESM model.")

    # Dataset & Dataloader
    train_dataset = AMP_Dataset(path=path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    model = TranCNN(d_output=1).to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for TranCNN model.")
    else:
        logging.info("Using single GPU for TranCNN model.")

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            AAI_embed = batch["aai"].squeeze(1).to(device)
            PAAC_embed = batch["paac"].squeeze(1).to(device)
            BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
            OH_embed = batch["onehot"].squeeze(1).to(device)
            labels = batch["label"].to(device)
            sequences = batch["sequence"]

            # -------- ESM Forward Pass -------- #
            with torch.no_grad():
                inputs = esm_tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=100,
                    add_special_tokens=False
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                torch.cuda.synchronize()
                start_esm = time.time()
                outputs = esm_model(**inputs)
                torch.cuda.synchronize()
                end_esm = time.time()

                esm_feat = outputs.states  # (B, L, D)
                esm_feat = esm_feat.transpose(0, 1)  # match model input shape

            logging.info(f"[Epoch {epoch+1}, Step {i+1}] ESM forward pass took {end_esm - start_esm:.4f} seconds")

            # -------- TranCNN Forward Pass -------- #
            torch.cuda.synchronize()
            start_trancnn = time.time()
            output = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed, esm_feat)
            torch.cuda.synchronize()
            end_trancnn = time.time()

            logging.info(f"[Epoch {epoch+1}, Step {i+1}] TranCNN forward pass took {end_trancnn - start_trancnn:.4f} seconds")

            # -------- Loss and Optimization -------- #
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('train/batch_loss', loss.item(), epoch * len(train_loader) + i)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 10 == 0:
            os.makedirs(save_model_dir, exist_ok=True)
            model_path = os.path.join(save_model_dir, f'TranCNN{epoch}.pth.tar')
            torch.save({'state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()}, model_path)
            logging.info(f"Model saved at {model_path}")

    writer.close()
    logging.info("Training completed.")

if __name__ == "__main__":
    train(path="./data/s1_train_")
