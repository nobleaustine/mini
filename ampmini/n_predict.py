
import sys
sys.path.append("./")
import os
from Bio import SeqIO
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
import scipy.stats
import pathlib
import copy
import time
# from termcolor import colored
import vocab 
from architecture import SequenceMultiTypeMultiCNN_1
from tools import EarlyStopping
from data_embed import Dataset
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse

all_function_names=['antibacterial','antigram-positive','antigram-negative','antifungal','antiviral',
                   'anti_mammalian_cells','antihiv','antibiofilm','anticancer','antimrsa','antiparasitic',
                   'hemolytic','chemotactic','antitb','anurandefense','cytotoxic',
                    'endotoxin','insecticidal','antimalarial','anticandida','antiplasmodial','antiprotozoal']

def return_y(data_iter, net):

    y_pred=[]
    all_seq=[]

    for batch in data_iter:

        all_seq+=batch['sequence']
        AAI_feat = batch['seq_enc_AAI'].to(device)
        onehot_feat = batch['seq_enc_onehot'].to(device)
        BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
        PAAC_feat = batch['seq_enc_PAAC'].to(device)
        
        outputs=net(AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat)

        y_pred.extend(outputs.cpu().numpy())


    return y_pred,all_seq

def testing(batch_size, testfasta, seq_len, model_file, device):
    model = SequenceMultiTypeMultiCNN_1(d_input=[531, 21, 23, 3], vocab_size=21, seq_len=seq_len,
                                        dropout=0.1, d_another_h=128, k_cnn=[2, 3, 4, 5, 6], d_output=1).to(device)
    
    dataset = Dataset(fasta=testfasta)
    test_loader = dataset.get_dataloader(batch_size=batch_size, max_length=seq_len)
    
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    
    with torch.no_grad():
        y_pred, all_seq = return_y(test_loader, model)
    
    y_pred = np.array(y_pred).T[0].tolist()
    return y_pred, all_seq

def predict(test_file):
    
    
    fas_id=[]
    fas_seq=[]
    labels=[]
    for seq_record in SeqIO.parse(os.path.join(test_file,"dummy_test_AMP.fasta"), "fasta"):
        fas_seq.append(str(seq_record.seq).upper())
        fas_id.append(str(seq_record.id))
        labels.append(1)
    for seq_record in SeqIO.parse(os.path.join(test_file,"dummy_test_NONAMP.fasta"), "fasta"):
        fas_seq.append(str(seq_record.seq).upper())
        fas_id.append(str(seq_record.id))
        labels.append(0)

       
    seq_len=200
    batch_size=64
    # cdhit_value=40
    # vocab_size = len(vocab.AMINO_ACIDS)

    model_file="./models_n/SequenceMultiTypeMultiCNN_1/AMP_1st/textcnn_cdhit_40_90.pth.tar"
    # epochs=300
    temp_save_AMP_filename='%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime()))
    pred_prob, _ = testing(batch_size=batch_size, testfasta=fas_seq, seq_len=seq_len, model_file=model_file, device=device)
    pred_labels = [1 if prob > 0.5 else 0 for prob in pred_prob]

    # pred_prob=[]
    # for cv_number in range(10):
    #     df=pd.read_csv(f'tmp_save/{temp_save_AMP_filename}_{cv_number}.csv')
    #     data=df.values.tolist()
    #     temp=[]
    #     for i in range(len(data)):
    #         temp.append(data[i][1])
    #     pred_prob.append(temp)

    # pred_prob=np.average(pred_prob,0)
    # pred_AMP_label=[]
    # for i in range(len(pred_prob)):
    #     if pred_prob[i]>0.5:
    #         pred_AMP_label.append(1)
    #     else:
    #         pred_AMP_label.append(0)


    # Ground truth labels
    y_true = labels

    # Predicted labels (binary after thresholding)
    y_pred = pred_labels

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    

    # for function_name in all_function_names:
    #     temp_dir_list=os.listdir('tmp_save')
    #     if function_name not in temp_dir_list:
    #         os.mkdir('tmp_save/'+function_name)
    #     for cv_number in range(10):
    #         testing(testfasta=fas_seq,
    #                 model_file=f'models/AMP_2nd/{function_name}/textcnn_cdhit_100_{cv_number}.pth.tar',
    #                 save_file=f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv',
    #                 batch_size=batch_size, patience=10, n_epochs=epochs,seq_len=seq_len,cdhit_value=cdhit_value,cv_number=cv_number)
    
    
    # all_function_pred_label=[]
    # for function_name in all_function_names:
        
    #     function_threshold_df=pd.read_csv(f'models/AMP_2nd_threashold/{function_name}_yd_threshold.csv',index_col=0)
    #     function_thresholds=function_threshold_df.values[:,0]       
        
    #     each_function_data=[]
        
    #     for cv_number in range(10):
    #         df=pd.read_csv(f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv')
    #         data=df.values.tolist()
    #         temp=[]
    #         for i in range(len(data)):

    #             if data[i][1]>function_thresholds[cv_number]:
    #                 temp.append(1)
    #             else:
    #                 temp.append(0)
    #         each_function_data.append(temp)
    #     each_function_data=np.average(each_function_data,0)
    #     pred_each_function_label=[]
    #     for i in range(len(each_function_data)):
    #         if each_function_data[i]>0.5:
    #             pred_each_function_label.append('Yes')
    #         else:
    #             pred_each_function_label.append('No')
                
    #     all_function_pred_label.append(pred_each_function_label)
        
    # all_function_cols=['antibacterial','anti-Gram-positive','anti-Gram-negative','antifungal','antiviral',\
    #                    'anti-mammalian-cells','anti-HIV','antibiofilm','anticancer','anti-MRSA','antiparasitic',\
    #                    'hemolytic','chemotactic','anti-TB','anurandefense','cytotoxic',\
    #                     'endotoxin','insecticidal','antimalarial','anticandida','antiplasmodial','antiprotozoal']

    
    # pred_contents_dict={'name':fas_id,'sequence':fas_seq,'AMP':pred_AMP_label}
    # for i in range(len(all_function_cols)):
    #     pred_contents_dict[all_function_cols[i]]=all_function_pred_label[i]
    
    
    # pred_contents_df=pd.DataFrame(pred_contents_dict)
    
    # for function_name in all_function_names:
    #     for cv_number in range(10):
    #         os.remove(f'tmp_save/{function_name}/{temp_save_AMP_filename}_{cv_number}.csv')
    # for cv_number in range(10):
    #     os.remove(f'tmp_save/{temp_save_AMP_filename}_{cv_number}.csv')        
    
    # return pred_contents_df
    # master.insert_one({'Test Report': res_val})
 
if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    parser = argparse.ArgumentParser(description='proposed model')
    parser.add_argument('-test_fasta_file', default='./data/AMP_1stage/', type=str)   
    args = parser.parse_args()

    test_file=args.test_fasta_file
    flag=0
    for seq_record in SeqIO.parse(os.path.join(test_file,"dummy_test_AMP.fasta"), "fasta"):
        temp_id=str(seq_record.id)
        temp_seq=str(seq_record.seq)
        if len(set(temp_seq.upper()).difference(set('ACDEFGHIKLMNPQRSTVWY')))>0:
            flag=1
            print('input error: have unusual amino acids')
            break
    for seq_record in SeqIO.parse(os.path.join(test_file,"dummy_test_NONAMP.fasta"), "fasta"):
        temp_id=str(seq_record.id)
        temp_seq=str(seq_record.seq)
        if len(set(temp_seq.upper()).difference(set('ACDEFGHIKLMNPQRSTVWY')))>0:
            flag=1
            print('input error: have unusual amino acids')
            break
    if flag==0:
        pred_df=predict(test_file)
