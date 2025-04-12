import sys
sys.path.append("./")
import torch
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
from network import SequenceTransformer,TranCNNnew
from AMP_dataset import AMP_Dataset
from collections import OrderedDict


# Setup logging
log_dir = "logs/SequenceTransformer/AMP_s2"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def evaluate(model_path, test_data_path, batch_size=64, num_classes=14):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Evaluating on {device}")
    
    # Load model
    model = TranCNNnew().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] 
    # model.load_state_dict(checkpoint['state_dict'])
    # If model was trained with DataParallel, keys will start with 'module.'
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())

    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.eval()
    logging.info(f"Loaded model from {model_path}")
    
    # Load test dataset
    test_dataset = AMP_Dataset(csv_path=test_data_path,embed_path = "./data/esmfold_test_features/")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logging.info("Test dataset loaded...")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            AAI_embed = batch["aai"].squeeze(1).to(device)
            PAAC_embed = batch["paac"].squeeze(1).to(device)
            BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
            OH_embed = batch["onehot"].squeeze(1).to(device)
            labels = batch["label"].squeeze(1).to(device)
            esm_feat = batch["esm_states"].squeeze(1).to(device)
            
            # x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]
            # outputs = model(x)
            outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed,esm_feat)
            probs = torch.sigmoid(outputs).cpu().numpy()
            # probs = probs.cpu().numpy()
            preds = (probs > 0.6).astype(int)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)
            print("labels", all_labels[-1][0])
            print("preds", all_preds[-1][0])
            print("probs", all_probs[-1][0])
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics for each class
    f1_scores = []
    mcc_scores = []
    auc_scores = []
    
    for i in range(num_classes):
        f1 = f1_score(all_labels[:, i], all_preds[:, i])
        mcc = matthews_corrcoef(all_labels[:, i], all_preds[:, i])
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            auc = float('nan')  # Handle cases where AUC cannot be computed
        
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        auc_scores.append(auc)
        logging.info(f"Class {i}: F1 = {f1:.4f}, MCC = {mcc:.4f}, AUC = {auc:.4f}")
    
    logging.info(f"Average F1-score: {np.mean(f1_scores):.4f}")
    logging.info(f"Average MCC: {np.mean(mcc_scores):.4f}")
    logging.info(f"Average AUC: {np.nanmean(auc_scores):.4f}")
    
if __name__ == "__main__":
    evaluate(model_path="./n_models/out/newTCNN80.pth.tar", test_data_path="./data/new_AMP_sequences_test.csv")
