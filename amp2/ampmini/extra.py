# from abc import ABC, abstractmethod

# class AMPD(Dataset, ABC):
#     def __init__(self, max_len=100, path="./data/embed_info"):
#         super().__init__()
#         self.max_len = max_len
#         self.blosum_dict = self._load_blosum(f"{path}/blosum62.txt")
#         self.aai_dict = self._load_aai(f"{path}/AAindex.txt")
#         self.paac_dict = self._load_paac(f"{path}/PAAC.txt")

#     @abstractmethod
#     def __len__(self):
#         pass

#     @abstractmethod
#     def __getitem__(self, idx):
#         pass

#     def _load_aai(self, path):
#         """Loads the AAindex data from a file and returns a dictionary."""
#         aai_dict = {}
#         with open(path) as f:
#             lines = [line.strip() for line in f if line.strip()]
        
#         header = lines[0].split('\t')[1:]  # Skip the first empty field
#         matrix = []
        
#         for line in lines[1:]:
#             values = line.split('\t')[1:]  # Skip the AA label
#             matrix.append([float(v) for v in values])
        
#         mat = np.array(matrix).T  # shape: (531, 21) -> columns are AAs
        
#         for i, aa in enumerate(header):
#             aai_dict[aa] = mat[i]
        
#         # Optional fallback
#         aai_dict['X'] = np.zeros(mat.shape[0], dtype=float)
        
#         return aai_dict

#     def _load_blosum(self, path):
#         """Loads the BLOSUM data from a file and returns a dictionary."""
#         blosum_dict = {}
#         print("path", path)
#         with open(path) as f:
#             lines = [line.strip() for line in f if line.strip()]
#         print("lines", lines)
#         header = lines[0].split()
#         print("header", header)
#         print("len header", len(header))

#         # Convert rows: skip first element (AA label), map the rest to float
#         matrix = [list(map(float, line.split()[1:])) for line in lines[1:]]
#         print("matrix", matrix)
#         print("len matrix", len(matrix))
#         print("len matrix[0]", len(matrix[0]))
#         mat = np.array(matrix)  # columns correspond to header AAs

#         for i, aa in enumerate(header):
#             blosum_dict[aa] = mat[i]
    
#         return blosum_dict
    
#     def _load_paac(self, path):
#         """Loads the PAAC data from a file and returns a dictionary."""
#         with open(path, 'r') as f:
#             text = f.read().split('\n')
        
#         text = [line for line in text if line.strip()]  # Remove empty lines
#         header = text[0].split('\t')
#         header = [item for item in header if item.strip()][1:]  # Skip the first column (AA label)
        
#         index = []
#         for line in text[1:]:
#             parts = line.split('\t')
#             parts = [item for item in parts if item.strip()][1:]  # Skip the first column (AA label)
#             index.append([float(x) for x in parts])
        
#         index = np.array(index)
        
#         paac_dict = {header[i]: index[:, i] for i in range(len(header))}
#         paac_dict['X'] = np.zeros(3)  # Fallback for unknown characters
        
#         return paac_dict
    
#     def ONEHOT_embed(self, seqs):
#         char_list = 'ARNDCQEGHILKMFPSTWYVX'
#         char_dict = {char: idx for idx, char in enumerate(char_list)}
        
#         all_embeds = []
#         for each_seq in seqs:
#             temp_embeds = []
#             for each_char in each_seq:
#                 codings = np.zeros(21)
#                 if each_char in char_dict:
#                     codings[char_dict[each_char]] = 1
#                 else:
#                     codings[20] = 1  # 'X' or unknown fallback
#                 temp_embeds.append(codings)
            
#             if len(each_seq) < self.max_len:
#                 zero_padding = np.zeros((self.max_len - len(each_seq), 21))
#                 data_pad = np.vstack((temp_embeds, zero_padding))
#             else:
#                 data_pad = temp_embeds[:self.max_len]
            
#             all_embeds.append(data_pad)

#         all_embeds = np.array(all_embeds)
#         return torch.from_numpy(all_embeds).float()  
    
#     def AAI_embed(self, seqs):
#         all_embeds = []
#         for seq in seqs:
#             temp_embeds = [
#                 self.aai_dict.get(aa, self.aai_dict['X']) for aa in seq
#             ]
#             if len(seq) < self.max_len:
#                 pad_len = self.max_len - len(seq)
#                 temp_embeds += [np.zeros_like(self.aai_dict['X'])] * pad_len
#             else:
#                 temp_embeds = temp_embeds[:self.max_len]
#             all_embeds.append(temp_embeds)
    
#         return torch.tensor(all_embeds, dtype=torch.float)
    
#     def BLOSUM62_embed(self, seqs):
#         all_embeds = []
#         for seq in seqs:
#             temp_embeds = [
#                 self.blosum_dict.get(aa, self.blosum_dict['X']) for aa in seq
#             ]
#             if len(seq) < self.max_len:
#                 pad_len = self.max_len - len(seq)
#                 temp_embeds += [np.zeros_like(self.blosum_dict['X'])] * pad_len
#             else:
#                 temp_embeds = temp_embeds[:self.max_len]
#             all_embeds.append(temp_embeds)
#         return torch.tensor(all_embeds, dtype=torch.float)
    
#     def PAAC_embed(self, seqs):
#         """Generate PAAC embeds for sequences."""
#         all_embeds = []
#         for each_seq in seqs:
#             temp_embeds = [self.paac_dict.get(each_char, self.paac_dict['X']) for each_char in each_seq]
            
#             if len(each_seq) < self.max_len:
#                 zero_padding = np.zeros((self.max_len - len(each_seq), 3))
#                 data_pad = np.vstack((temp_embeds, zero_padding))
#             else:
#                 data_pad = temp_embeds[:self.max_len]
            
#             all_embeds.append(data_pad)

#         all_embeds = np.array(all_embeds)
#         return torch.from_numpy(all_embeds).float()

# class AMPs2(AMPD):
#     def __init__(self, data_path="./data/s2_train"):
#         super().__init__(max_len=100, path="./data/embed_info")
#         data = pd.read_csv(f"{data_path}_AMPseqs.csv", sep=",")
#         self.ids = data["id"].values
#         self.data = {id: (seq, label) for id, seq, label in zip(
#             data["id"].values,
#             data["sequence"].values,
#             data.iloc[:, 2:].values.astype(float))}
#         self.embed_path = f"{data_path}_ESMfeats/"
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         id = self.ids[idx]
#         sequence,label = self.data[id]
#         esm_np = np.load(os.path.join(self.embed_path, f"{id}.npz"))
#         esmdata = {key: torch.from_numpy(esm_np[key]) for key in esm_np.files}

#         return {
#             "sequence": sequence,
#             "onehot": self.ONEHOT_embed([sequence]),
#             "blosum62": self.BLOSUM62_embed([sequence]),
#             "aai": self.AAI_embed([sequence]),
#             "paac": self.PAAC_embed([sequence]),
#             # "esm_s_s": esmdata["s_s"],
#             # "esm_s_z": esmdata["s_z"],
#             "esm_states": esmdata["states"],
#             "label": torch.tensor(label, dtype=torch.float32)
#         }
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class TranCNNold(nn.Module):

    def __init__(
        self, 
        d_input=[531,21,23,3],
        vocab_size=None, 
        seq_len=200,
        dropout=0.1,
        hid_dim=32,
        k_cnn=[2,3,4],
        d_output=24,
        k_size=2,
        depth = 4):
        super(TranCNN, self).__init__()
        
        self.batchnorm_4=nn.BatchNorm1d(num_features=d_input[3])
        self.convs_1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[0], 
                                        out_channels=hid_dim, 
                                        kernel_size=h,padding = h//2),
                              nn.BatchNorm1d(num_features=hid_dim), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=k_size))
                     for h in k_cnn
                    ])
        self.convs_2 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[1], 
                                        out_channels=hid_dim, 
                                        kernel_size=h,padding = h//2),
                              nn.BatchNorm1d(num_features=hid_dim), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=k_size))
                     for h in k_cnn
                    ])
        self.convs_3 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[2], 
                                        out_channels=hid_dim, 
                                        kernel_size=h,padding = h//2),
                              nn.BatchNorm1d(num_features=hid_dim), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=k_size))
                     for h in k_cnn
                    ])
        self.convs_4 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[3], 
                                        out_channels=hid_dim, 
                                        kernel_size=h,padding = h//2),
                              nn.BatchNorm1d(num_features=hid_dim), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=k_size))
                     for h in k_cnn
                    ])
        
        self.maxpool_1=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_2=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_3=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_4=nn.MaxPool1d(kernel_size=len(k_cnn))

        self.conv = nn.Sequential(nn.Conv1d(in_channels=4*hid_dim, 
                                        out_channels=2*hid_dim, 
                                        kernel_size=3, 
                                        padding=1),
                                        nn.BatchNorm1d(num_features=2*hid_dim),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2))
        
        # self.middle = nn.ModuleList([MBlock(2*hid_dim, 4, mlp_ratio=2.0, dropout=dropout) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.final = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=2*hid_dim, 
                                        out_channels=2*hid_dim, 
                                        kernel_size=3,padding = k//2),
                              nn.BatchNorm1d(num_features=2*hid_dim), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=k))
                     for k in [5,5,2]
                    ])
        
        self.fc = nn.Linear(2*hid_dim, d_output)
        self.sigmoid=nn.Sigmoid()

    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat):
        
        AAI_feat = AAI_feat.permute(0,2,1)
        out_1 = [conv(AAI_feat) for conv in self.convs_1] 
        onehot_feat = onehot_feat.permute(0,2,1)
        out_2 = [conv(onehot_feat) for conv in self.convs_2] 
        BLOSUM62_feat = BLOSUM62_feat.permute(0,2,1)
        out_3 = [conv(BLOSUM62_feat) for conv in self.convs_3] 
        PAAC_feat = PAAC_feat.permute(0,2,1)
        out_4 = [conv(PAAC_feat) for conv in self.convs_4] 
        

        out_1 = torch.cat(out_1, dim=2)
        out_1=self.maxpool_1(out_1)
        out_2 = torch.cat(out_2, dim=2)
        out_2=self.maxpool_2(out_2)
        out_3 = torch.cat(out_3, dim=2)
        out_3=self.maxpool_3(out_3)
        out_4 = torch.cat(out_4, dim=2)
        out_4=self.maxpool_4(out_4)
    
        x=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        x = self.conv(x)
        # x = x.permute(0,2,1)
        # for mlayer in self.middle:
        #     x = mlayer(x, x)
        # x = x.permute(0,2,1)
        for layer in self.final:
            x = layer(x)
        x = x.view(-1, x.size(1)) 
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class mMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, qk_scale=None,qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv_p = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,y):
        B, N, C = x.shape
        nh = self.num_heads
        head_dim = C // nh
        odd_indices = torch.arange(nh) % 2 == 1
        
        # Compute q, k, v for P
        qkv_p = self.qkv_p(x).reshape(B, N, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
        Pq, Pk, Pv = qkv_p[0], qkv_p[1], qkv_p[2]
        
        # Compute q, k, v for C
        qkv_c = self.qkv_c(y).reshape(B, N, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
        Cq, Ck, Cv = qkv_c[0], qkv_c[1], qkv_c[2]
        
        # Compute attention matrices
        qk_p = (Pq @ Pk.transpose(-2, -1)) * self.scale
        attn_p1 = F.softmax(qk_p.clone(),dim=-1)
        attn_p = self.attn_drop(attn_p1.clone())
        
        qk_c = (Cq @ Ck.transpose(-2, -1)) * self.scale
        attn_c1 = F.softmax(qk_c.clone(),dim=-1)
        attn_c = self.attn_drop(attn_c1.clone())
        
        # Swap odd-indexed heads between P and C
        attn_p[:, odd_indices, :, :], attn_c[:, odd_indices, :, :] = attn_c[:, odd_indices, :, :], attn_p[:, odd_indices, :, :]
        
        # Apply attention to values
        v1 = (attn_p @ Pv).transpose(1, 2).reshape(B, N, C)
        v2 = (attn_c @ Cv).transpose(1, 2).reshape(B, N, C)
        
        # Projection
        v1 = self.proj(v1)
        v1 = self.proj_drop(v1)
        
        v2 = self.proj(v2)
        v2 = self.proj_drop(v2)

        attn = torch.cat((attn_p, attn_c), dim=1)
        
        return v1, v2, attn


class SequenceMultiTypeMultiCNN_2(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
        super(SequenceMultiTypeMultiCNN_2, self).__init__()
        
        self.batchnorm_4=nn.BatchNorm1d(num_features=d_input[3])
        self.convs_1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[0], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_2 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[1], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_3 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[2], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.convs_4 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=d_input[3], 
                                        out_channels=d_another_h, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=d_another_h), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=seq_len-h+1))
                     for h in k_cnn
                    ])
        self.maxpool_1=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_2=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_3=nn.MaxPool1d(kernel_size=len(k_cnn))
        self.maxpool_4=nn.MaxPool1d(kernel_size=len(k_cnn))
        # self.maxpool_1=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_2=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_3=nn.AvgPool1d(kernel_size=5)
        # self.maxpool_4=nn.AvgPool1d(kernel_size=5)
        self.drop = nn.Dropout(dropout)
        
        self.fc_1 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_2 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_3 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc_4 = nn.Linear(d_another_h*len(k_cnn), d_output)
        self.fc = nn.Linear(4*d_another_h, d_output)
        self.sigmoid=nn.Sigmoid()



    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat):
        # print(x)
        AAI_feat = AAI_feat.permute(0,2,1)
        out_1 = [conv(AAI_feat) for conv in self.convs_1] 
        onehot_feat = onehot_feat.permute(0,2,1)
        out_2 = [conv(onehot_feat) for conv in self.convs_2] 
        BLOSUM62_feat = BLOSUM62_feat.permute(0,2,1)
        out_3 = [conv(BLOSUM62_feat) for conv in self.convs_3] 
        
        PAAC_feat = PAAC_feat.permute(0,2,1)
        PAAC_feat = self.batchnorm_4(PAAC_feat)
        out_4 = [conv(PAAC_feat) for conv in self.convs_4] 
        out_1 = torch.cat(out_1, dim=2)
        # print(out_1.size())
        out_1=self.maxpool_1(out_1)
        # print(out_1.size())
        out_2 = torch.cat(out_2, dim=2)
        out_2=self.maxpool_2(out_2)
        out_3 = torch.cat(out_3, dim=2)
        out_3=self.maxpool_3(out_3)
        out_4 = torch.cat(out_4, dim=2)
        out_4=self.maxpool_4(out_4)
        # print(out_4.size())
        x=torch.cat([out_1,out_2,out_3,out_4],dim=1)
        x = x.view(-1, x.size(1)) 
        # out=torch.cat([out_1,out_2,out_3,out_4], dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x,out_1,out_2,out_3,out_4

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=24, hidden_dim=256, dropout=0.3):
        super(MultiLabelClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Second hidden layer
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_out(x)
        x = torch.sigmoid(x)  # Sigmoid for multi-label classification
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class sBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = mMHSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(2*dim)
        mlp_hidden_dim = int(2*dim*mlp_ratio)
        self.mlp = Mlp(
            in_features=2*dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, y, return_attention=False):
        b1,b2 = x,y
        x, y, attn = self.attn(self.norm1(x),self.norm11(y))
        if return_attention:
            return attn
        x = b1 + self.drop_path(x)
        y = b2 + self.drop_path(y)
        dxy =  self.drop_path(self.mlp(self.norm2(torch.cat([x,y],dim=2))))
        x = x + dxy
        y = y + dxy
        return x,y

class Embed(nn.Module):
    """Sequence Embedding to Transformer Embedding"""

    def __init__(self, in_chans, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class SequenceTransformer(nn.Module):
   """AMP Sequence Transformer"""
    def __init__(
        self,
        in_chans=[531, 3, 23, 21],
        num_classes=24,
        embed_dim=128,
        num_tokens=200,
        depth=6,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.Ppatch = Embed(in_chans[0]+in_chans[1], embed_dim)
        self.Cpatch = Embed(in_chans[2]+in_chans[3], embed_dim)

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.sigmoid=nn.Sigmoid()

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                sBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(2*embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):

        AAI_embed= x[0]
        PAAC_embed= x[1]
        BLOSUM62_embed= x[2]
        OH_embed= x[3]
        
        B,T,D = AAI_embed.shape
        # print("AAI_embed",AAI_embed.shape)
        # print("PAAC_embed",PAAC_embed.shape)
        # print("BLOSUM62_embed",BLOSUM62_embed.shape)
        # print("OH_embed",OH_embed.shape)
        x = self.Ppatch(torch.cat([AAI_embed, PAAC_embed], dim=2))
        y = self.Cpatch(torch.cat([BLOSUM62_embed, OH_embed], dim=2)) 

        # add the [CLS] token to the embed patch tokens
        cls_token1 = self.cls_token1.expand(B, -1, -1)
        cls_token2 = self.cls_token2.expand(B, -1, -1)
        # print("cls_token1",cls_token1.shape)
        # print("cls_token2",cls_token2.shape)
        x = torch.cat((cls_token1, x), dim=1)
        y = torch.cat((cls_token2, y), dim=1)

        return self.pos_drop(x), self.pos_drop(y)

    def forward(self, x):
        x,y = self.prepare_tokens(x)
        for blk in self.blocks:
            x,y = blk(x,y)
        x = self.norm(x)
        y = self.norm(y)
        out = self.head(torch.cat((x[:, 0], y[:, 0]), dim=1))
        pro= self.sigmoid(out)
        return pro

    # def get_last_selfattention(self, x):
    #     x,y = self.prepare_tokens(x)
    #     for i, blk in enumerate(self.blocks):
    #         if i < len(self.blocks) - 1:
    #             x = blk(x)
    #         else:
    #             # return attention of the last block
    #             return blk(x, return_attention=True)

    # def get_intermediate_layers(self, x, n=1):
    #     x = self.prepare_tokens(x)
    #     # we return the output tokens from the `n` last blocks
    #     output = []
    #     for i, blk in enumerate(self.blocks):
    #         x = blk(x)
    #         if len(self.blocks) - i <= n:
    #             output.append(self.norm(x))
    #     return output

    # import sys
# sys.path.append("./")

# import os
# import logging
# import torch
# torch.backends.cudnn.benchmark = True  # Optimized for fast training
# import torch.optim as optim
# from torch.nn import DataParallel
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
# from network import SequenceTransformer, FocalLoss, TranCNN
# from AMP_dataset import AMP_Dataset

# # Setup logging
# # log_dir = "logs/TranCNN/AMP_s2/dataparallel"
# log_dir = "logs/ST/AMP_s2/dataparallel"
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, "log.txt")
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# def train(path="", batch_size=64, learning_rate=0.001, num_epochs=102, save_model_dir="n_models/ST/AMP_s2/dataparallel"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Training on {device} using DataParallel")

#     # Load dataset
#     train_dataset = AMP_Dataset(csv_path=path)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     logging.info("Dataset and Dataloader created...")

#     # Model & DataParallel setup
#     model = SequenceTransformer().to(device)
#     if torch.cuda.device_count() > 1:
#         model = DataParallel(model)
#         logging.info(f"Using {torch.cuda.device_count()} GPUs!")

#     criterion = FocalLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

#     writer = SummaryWriter(log_dir)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0

#         for i, batch in enumerate(train_loader):
#             AAI_embed = batch["aai"].squeeze(1).to(device)
#             PAAC_embed = batch["paac"].squeeze(1).to(device)
#             BLOSUM62_embed = batch["blosum62"].squeeze(1).to(device)
#             OH_embed = batch["onehot"].squeeze(1).to(device)
#             labels = batch["label"].squeeze(1).to(device)
#             x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]

#             # outputs = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed)
#             outputs = model(x)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if (i + 1) % 100 == 0:
#                 msg = f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
#                 logging.info(msg)
#                 writer.add_scalar('train/training loss', loss.item(), epoch * len(train_loader) + i)

#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader)
#         logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
#         writer.add_scalar('train/epoch loss', epoch_loss, epoch)
#         writer.add_scalar('train/learning rate', scheduler.get_last_lr()[0], epoch)

#         # Save model checkpoint
#         os.makedirs(save_model_dir, exist_ok=True)
#         if epoch % 10 == 0:
#             model_path = os.path.join(save_model_dir, f'TCN{epoch}.pth.tar')
#             torch.save({'state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()}, model_path)
#             logging.info(f"Model saved at {model_path}")
#     writer.close()
#     logging.info("Training completed.")

# if __name__ == "__main__":
#     train(path="./data/AMP_sequences.csv")