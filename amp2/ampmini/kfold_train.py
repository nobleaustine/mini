import sys
sys.path.append("./")

import os
import time
import logging
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from datetime import datetime
from network import FocalLoss, CTC
from AMP_Dataset import AMP_Dataset


def train_kfold(path="./data/s2_train_", 
                batch_size=64, 
                learning_rate=0.001, 
                num_epochs=50, 
                save_model_dir="models/s2AMP/CTC/",
                multi_gpu=True,
                esm_flag=True,
                model_name="CTC",
                log_dir="logs/s2AMP/CTC/",
                k_folds=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device} with {k_folds}-Fold Cross-Validation")

    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    full_dataset = AMP_Dataset(path=path, esm_flag=esm_flag)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        logging.info(f"===== Fold {fold + 1} / {k_folds} =====")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # Model setup
        model = CTC(out_dim=1, use_attn=True).to(device)
        if torch.cuda.device_count() >= 1 and multi_gpu:
            model = DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")

        criterion = FocalLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        writer = SummaryWriter(os.path.join(log_dir, f"fold_{fold+1}"))

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, batch in enumerate(train_loader):
                AAI_embed = batch["aai"].squeeze(1).to(device)
                PAAC_embed = batch["paac"].squeeze(1).to(device)
                BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
                OH_embed = batch["onehot"].squeeze(1).to(device)
                labels = batch["label"].to(device)
                esm_feat = batch["esm_states"].to(device)

                outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed, esm_feat)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('train/batch_loss', loss.item(), global_step)

            epoch_loss = running_loss / len(train_loader)
            scheduler.step()

            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(save_model_dir, f"{model_name}_fold{fold+1}_epoch{epoch+1}.pth.tar")
                torch.save({'state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()}, ckpt_path)
                logging.info(f"Checkpoint saved: {ckpt_path}")

        # Optional: simple validation loop after training
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                AAI_embed = batch["aai"].squeeze(1).to(device)
                PAAC_embed = batch["paac"].squeeze(1).to(device)
                BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
                OH_embed = batch["onehot"].squeeze(1).to(device)
                labels = batch["label"].to(device)
                esm_feat = batch["esm_states"].to(device)

                outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed, esm_feat)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("val/final_loss", val_loss, 0)
        logging.info(f"Fold {fold+1} validation loss: {val_loss:.4f}")

        writer.close()

    logging.info("K-Fold Training Completed.")


if __name__ == "__main__":
    log_dir = "../logs/s2AMP/CTCk/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "kfold_train_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    train_kfold(
        path="./data/s2_train_",
        batch_size=512,
        learning_rate=0.001,
        num_epochs=50,
        save_model_dir="./models/s2AMP/CTCk/",
        multi_gpu=True,
        esm_flag=True,
        model_name="CTC",
        log_dir=log_dir,
        k_folds=5
    )
