# import sys
# sys.path.append("./")
# import torch
# import torch.nn.functional as F
# import logging
# import os
# import sys
# import numpy as np
# from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
# from torch.utils.data import DataLoader
# from network import CTC  #CNN_1 # TranCNN,SequenceMultiTypeMultiCNN_1
# from AMP_Dataset import AMP_Dataset
# from collections import OrderedDict

# def evaluate(model_path, 
#         test_data_path, 
#         batch_size=64, 
#         num_cls=1,
#         esm_flag=True,
#         log_dir=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Evaluating on {device}")
    
#     # Load model
#     # model = TranCNN(d_output=14).to(device)
#     # model = SequenceMultiTypeMultiCNN_1(d_output=num_cls).to(device)
#     # model = CNN_1(d_output=num_cls).to(device)
#     model = CTC(out_dim=num_cls,use_attn=False).to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     state_dict = checkpoint['state_dict'] 
#     # model.load_state_dict(checkpoint['state_dict'])
#     # If model was trained with DataParallel, keys will start with 'module.'
#     if any(k.startswith('module.') for k in state_dict.keys()):
#         # Remove 'module.' prefix
#         new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())

#     else:
#         new_state_dict = state_dict

#     model.load_state_dict(new_state_dict)
#     model.eval()
#     logging.info(f"Loaded model from {model_path}")
    
#     # Load test dataset
#     test_dataset = AMP_Dataset(path=test_data_path, esm_flag=esm_flag)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     logging.info("Test dataset loaded...")
    
#     all_labels = []
#     all_preds = []
#     all_probs = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             AAI_embed = batch["aai"].squeeze(1).to(device)
#             PAAC_embed = batch["paac"].squeeze(1).to(device)
#             BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
#             OH_embed = batch["onehot"].squeeze(1).to(device)
#             labels = batch["label"].to(device) #.squeeze(1)
#             esm_feat = batch["esm_states"].squeeze(1).to(device)
            
#             # x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]
#             # outputs = model(x)
#             outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed,esm_feat)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             # probs = probs.cpu().numpy()
#             preds = (probs > 0.6).astype(int)
#             # if preds.shape == (64,1):
#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds)
#             all_probs.append(probs)
#             # print(labels.shape)
#             # print(preds.shape)
#             print("labels", all_labels[-1][0])
#             print("preds", all_preds[-1][0])
#             print("probs", all_probs[-1][0])
#             # else:
#             #     print(preds.shape)
    
#     all_labels = np.vstack(all_labels)
#     all_preds = np.vstack(all_preds)
#     all_probs = np.vstack(all_probs)
#     print("all_labels", all_labels.shape)
#     print("all_preds", all_preds.shape)
#     print("all_probs", all_probs.shape)
    
#     # Calculate metrics for each class
#     f1_scores = []
#     mcc_scores = []
#     auc_scores = []
    
#     for i in range(num_cls):
#         f1 = f1_score(all_labels[:, i], all_preds[:, i])
#         mcc = matthews_corrcoef(all_labels[:, i], all_preds[:, i])
#         try:
#             auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
#         except ValueError:
#             auc = float('nan')  # Handle cases where AUC cannot be computed
        
#         f1_scores.append(f1)
#         mcc_scores.append(mcc)
#         auc_scores.append(auc)
#         logging.info(f"Class {i}: F1 = {f1:.4f}, MCC = {mcc:.4f}, AUC = {auc:.4f}")
    
#     logging.info(f"Average F1-score: {np.mean(f1_scores):.4f}")
#     logging.info(f"Average MCC: {np.mean(mcc_scores):.4f}")
#     logging.info(f"Average AUC: {np.nanmean(auc_scores):.4f}")
    
# if __name__ == "__main__":

#     # Setup logging
#     log_dir = "logs/s2AMP/CTC/"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, "1test_log.txt")
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )

#     evaluate("./models/s2AMP/CTC/CTC100.pth.tar", # "./models/s2AMP/TranCNN/TranCNN80.pth.tar" 
#         "./data/s2_test_", 
#         batch_size=64, 
#         num_cls=14,
#         esm_flag=False,
#         log_dir=log_dir)
   

# def evaluate(model_path, 
#         test_data_path, 
#         batch_size=64, 
#         num_cls=1,
#         esm_flag=True,
#         log_dir=None,
#         threshold_range=(0.1, 0.9),
#         threshold_step=0.05):
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Evaluating on {device}")

#     model = CTC(out_dim=num_cls, use_attn=True).to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     state_dict = checkpoint['state_dict']
    
#     if any(k.startswith('module.') for k in state_dict.keys()):
#         new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
#     else:
#         new_state_dict = state_dict
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     logging.info(f"Loaded model from {model_path}")
    
#     test_dataset = AMP_Dataset(path=test_data_path, esm_flag=esm_flag)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     logging.info("Test dataset loaded...")

#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in test_loader:
#             AAI_embed = batch["aai"].squeeze(1).to(device)
#             PAAC_embed = batch["paac"].squeeze(1).to(device)
#             BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
#             OH_embed = batch["onehot"].squeeze(1).to(device)
#             labels = batch["label"].to(device)
#             esm_feat = batch["esm_states"].squeeze(1).to(device)

#             outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed, esm_feat)
#             probs = torch.sigmoid(outputs).cpu().numpy()
#             all_labels.append(labels.cpu().numpy())
#             all_probs.append(probs)

#     all_labels = np.vstack(all_labels)
#     all_probs = np.vstack(all_probs)

#     logging.info(f"all_labels shape: {all_labels.shape}")
#     logging.info(f"all_probs shape: {all_probs.shape}")

#     # Threshold search
#     best_threshold = None
#     best_f1 = -1
#     thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)

#     for t in thresholds:
#         preds = (all_probs > t).astype(int)
#         f1_scores = [f1_score(all_labels[:, i], preds[:, i]) for i in range(num_cls)]
#         avg_f1 = np.mean(f1_scores)
#         logging.info(f"Threshold {t:.2f} → Avg F1: {avg_f1:.4f}")
#         if avg_f1 > best_f1:
#             best_f1 = avg_f1
#             best_threshold = t

#     logging.info(f"Best threshold: {best_threshold:.2f} with Avg F1: {best_f1:.4f}")

#     # Compute final metrics at best threshold
#     final_preds = (all_probs > best_threshold).astype(int)
#     f1_scores = []
#     mcc_scores = []
#     auc_scores = []

#     for i in range(num_cls):
#         f1 = f1_score(all_labels[:, i], final_preds[:, i])
#         mcc = matthews_corrcoef(all_labels[:, i], final_preds[:, i])
#         try:
#             auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
#         except ValueError:
#             auc = float('nan')
        
#         f1_scores.append(f1)
#         mcc_scores.append(mcc)
#         auc_scores.append(auc)
#         logging.info(f"Class {i}: F1 = {f1:.4f}, MCC = {mcc:.4f}, AUC = {auc:.4f}")

#     logging.info(f"Final Avg F1-score: {np.mean(f1_scores):.4f}")
#     logging.info(f"Final Avg MCC: {np.mean(mcc_scores):.4f}")
#     logging.info(f"Final Avg AUC: {np.nanmean(auc_scores):.4f}")

# if __name__ == "__main__":
#     log_dir = "logs/s2AMP/CTC/"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, "1test_log.txt")
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )

#     evaluate(
#         "./models/s2AMP/CTCk/CTC_fold1_epoch40.pth.tar",
#         "./data/s2_test_", 
#         batch_size=128, 
#         num_cls=1,
#         esm_flag=True,
#         log_dir=log_dir,
#         threshold_range=(0.1, 0.9),
#         threshold_step=0.05
#    )

import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from network import CTC
from AMP_Dataset import AMP_Dataset

def load_model(model_path, device, num_cls=1):
    model = CTC(out_dim=num_cls, use_attn=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_predictions(model, dataloader, device):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            AAI_embed = batch["aai"].squeeze(1).to(device)
            PAAC_embed = batch["paac"].squeeze(1).to(device)
            BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
            OH_embed = batch["onehot"].squeeze(1).to(device)
            labels = batch["label"].to(device)
            esm_feat = batch["esm_states"].squeeze(1).to(device)

            outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed, esm_feat)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    return np.vstack(all_labels), np.vstack(all_probs)

def find_best_threshold(labels, probs, threshold_range=(0.1, 0.9), step=0.05):
    best_threshold, best_f1 = 0.5, -1
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1

def evaluate_all_folds(model_dir, test_path, fold_nums, batch_size=128, esm_flag=True, threshold_csv="thresholds.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = AMP_Dataset(path=test_path, esm_flag=esm_flag)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    thresholds = []
    all_fold_probs = []

    for fold in fold_nums:
        model_path = os.path.join(model_dir, f"CTC_fold{fold}_epoch50.pth.tar")
        model = load_model(model_path, device)
        labels, probs = get_predictions(model, test_loader, device)

        threshold, f1 = find_best_threshold(labels, probs)
        print(f"[Fold {fold}] Best threshold: {threshold:.2f}, F1: {f1:.4f}")
        thresholds.append({"fold": fold, "threshold": threshold})
        all_fold_probs.append(probs)

    pd.DataFrame(thresholds).to_csv(threshold_csv, index=False)
    print(f"Saved thresholds to {threshold_csv}")

    return labels, all_fold_probs, thresholds

def ensemble_and_evaluate(labels, all_probs, thresholds):
    avg_probs = np.mean(np.stack(all_probs), axis=0)
    avg_preds = (avg_probs > 0.5).astype(int)

    f1 = f1_score(labels, avg_preds)
    mcc = matthews_corrcoef(labels, avg_preds)
    try:
        auc = roc_auc_score(labels, avg_probs)
    except ValueError:
        auc = float("nan")

    print(f"Ensemble Evaluation @ threshold=0.5 --> F1: {f1:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    model_dir = "./models/s2AMP/CTCk/"
    test_path = "./data/s2_test_"
    fold_nums = [1, 2, 3, 4, 5]
    threshold_csv = "./models/s2AMP/CTCk/thresholds.csv"

    labels, all_probs, thresholds = evaluate_all_folds(
        model_dir=model_dir,
        test_path=test_path,
        fold_nums=fold_nums,
        batch_size=128,
        esm_flag=True,
        threshold_csv=threshold_csv
    )

    ensemble_and_evaluate(labels, all_probs, thresholds)


