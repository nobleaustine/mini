import sys
sys.path.append("./")

import os
import time
import logging
import torch
torch.backends.cudnn.benchmark = True  # Optimized for fast training
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from network import  FocalLoss,CTC  # CNN_1,TranCNN, SequenceMultiTypeMultiCNN_1
from AMP_Dataset import AMP_Dataset


def train(path="./data/s2_train_", 
          batch_size=64, 
          learning_rate=0.001, 
          num_epochs=102, 
          save_model_dir="models/s2AMP/TranCNN/",
          multi_gpu=True,
          esm_flag=True,
          model_name="TranCNN",
          log_dir="logs/s2AMP/TranCNN/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device} using DataParallel")

    # ⏱ Dataset Loading
    start_time = time.time()
    train_dataset = AMP_Dataset(path=path,esm_flag=esm_flag)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    logging.info(f"Dataset and Dataloader created in {time.time() - start_time:.2f} seconds.")

    # ⏱ Model Setup
    start_time = time.time()
    # model = TranCNN(d_output=14).to(device)
    # model = SequenceMultiTypeMultiCNN_1().to(device) # change model here
    # model = CNN_1(d_output=1).to(device)
    model = CTC(out_dim=14,use_attn=True).to(device)
    if torch.cuda.device_count() >= 1 and multi_gpu:
        model = DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    logging.info(f"Model setup completed in {time.time() - start_time:.2f} seconds.")

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        total_dataload_time = 0.0
        total_forward_time = 0.0
        total_loss_time = 0.0
        total_backward_time = 0.0

        for i, batch in enumerate(train_loader):
            
            batch_start = time.time()

            AAI_embed = batch["aai"].squeeze(1).to(device)
            PAAC_embed = batch["paac"].squeeze(1).to(device)
            BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
            OH_embed = batch["onehot"].squeeze(1).to(device)
            labels = batch["label"].to(device) #.squeeze(1)     
            esm_feat = batch["esm_states"].to(device)
            # x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]
            # outputs = model(x)

            # ⏱ Forward
            forward_start = time.time()
            outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed,esm_feat)
            forward_end = time.time()
            total_forward_time += forward_end - forward_start

            # ⏱ Loss
            loss_start = time.time()
            loss = criterion(outputs, labels)
            loss_end = time.time()
            total_loss_time += loss_end - loss_start

            # ⏱ Backward
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_end = time.time()
            total_backward_time += backward_end - backward_start

            running_loss += loss.item()
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/batch_loss', loss.item(), global_step)

            if (i + 1) % 100 == 0:
                batch_time = time.time() - batch_start
                msg = f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f}s"
                logging.info(msg)


        # addup = time.time()
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ⏱ Profiling Summary
        dataload_pct = (total_dataload_time / epoch_time) * 100
        forward_pct = (total_forward_time / epoch_time) * 100
        loss_pct = (total_loss_time / epoch_time) * 100
        backward_pct = (total_backward_time / epoch_time) * 100
        misc_pct = 100 - (dataload_pct + forward_pct + loss_pct + backward_pct)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f}s")
        logging.info(f"Time Breakdown (% of epoch): DataLoad: {dataload_pct:.2f}%, Forward: {forward_pct:.2f}%, Loss: {loss_pct:.2f}%, Backward: {backward_pct:.2f}%, Misc: {misc_pct:.2f}%")
        logging.info(f"Raw Time (s): DataLoad: {total_dataload_time:.2f}, Forward: {total_forward_time:.2f}, Loss: {total_loss_time:.2f}, Backward: {total_backward_time:.2f}, Total: {epoch_time:.2f}")

        # writer.add_scalar('train/epoch loss', epoch_loss, epoch)
        # writer.add_scalar('train/learning rate', scheduler.get_last_lr()[0], epoch)

        # Save model checkpoint
        os.makedirs(save_model_dir, exist_ok=True)
        if epoch % 10 == 0:
            model_path = os.path.join(save_model_dir, f'{model_name}{epoch}.pth.tar')
            torch.save({'state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()}, model_path)
            logging.info(f"Model saved at {model_path}")
        # extra = time.time() - addup
        # print(extra)

    writer.close()
    logging.info("Training completed.")

if __name__ == "__main__":
    

    # log_dir = "logs/s2AMP/TranCNN/"
    log_dir = "../logs/s2AMP/CTC/"
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
    train(path="./data/s2_train_", 
          batch_size=64, 
          learning_rate=0.001, 
          num_epochs=102, 
          save_model_dir="../models/s1AMP/CTC/",
          multi_gpu=True,
          esm_flag=False,
          model_name="CTC",
          log_dir=log_dir)
