import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class MBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3.0, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadCrossAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context,pos_embed):
        # Cross-attention
        attn_out = self.cross_attn(x, context,pos_embed)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)

        return x

class TranCNN(nn.Module):
    def __init__(
        self, 
        d_input=[531,21,23,3],
        vocab_size=21, 
        seq_len=100,
        dropout=0.2,
        hid_dim=64,
        k_cnn=[2,3,4],
        d_output=14,
        k_size=2,
        depth=6,
        attn_dim=384):
        super(TranCNN, self).__init__()


        self.convs_all = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=sum(d_input), out_channels=hid_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=k_size)
            ) for k in k_cnn
        ])

        self.maxpool = nn.MaxPool1d(kernel_size=len(k_cnn))
        self.conv_embed = nn.Conv1d(in_channels=hid_dim, out_channels=attn_dim, kernel_size=1,stride = 1,padding=0)

        self.middle = nn.ModuleList([
            MBlock(attn_dim, 4, mlp_ratio=2.0, dropout=dropout) for _ in range(depth)
        ])
        self.conv_back = nn.Conv1d(in_channels=attn_dim, out_channels=hid_dim, kernel_size=1,stride= 1,padding=0)
        self.final = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=k//2),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=k)
            ) for k in [5, 5, 2]
        ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat, esm_feat):
        x = torch.cat([AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat], dim=2)
        x = x.permute(0, 2, 1)

        out = [conv(x) for conv in self.convs_all]
        x = torch.cat(out, dim=2)
        x = self.maxpool(x)
        x = self.conv_embed(x)
        x = x.permute(0, 2, 1)

        pos_embed = get_sinusoid_encoding_table(x.size(1), x.size(2), device=x.device)
        x = x + pos_embed.unsqueeze(0)

        i = 0
        for block in self.middle:
            x = block(x, esm_feat[:,2+i,...], pos_embed)
            i += 1

        x = self.conv_back(x.permute(0, 2, 1))
        for layer in self.final:
            x = layer(x)

        x = x.mean(dim=2)
        x = self.drop(x)
        x = self.fc(x)
        return self.sigmoid(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        """
        Implements Focal Loss.
        
        Args:
            alpha (float): Balancing factor (default: 1).
            gamma (float): Focusing parameter (default: 2).
            logits (bool): If True, expects raw logits as input.
            reduction (str): 'mean' or 'sum' or 'none' (default: 'mean').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            prob = torch.sigmoid(inputs)  # Compute probability from logits
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            prob = inputs  # Probabilities are already in range [0,1]

        pt = prob * targets + (1 - prob) * (1 - targets)  # Correct calculation of pt
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  # 'none' reduction

class SequenceMultiTypeMultiCNN_1(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, 
                seq_len=100,
                dropout=0.1,
                d_another_h=64,
                k_cnn=[2,3,4,5,6],
                d_output=1):
        super(SequenceMultiTypeMultiCNN_1, self).__init__()
        
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

    def forward(self, AAI_feat,onehot_feat,BLOSUM62_feat,PAAC_feat,esm_feat):
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
        return x

def sinusoid_pos_encod(seq_len, dim):
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1,seq_len=50):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.pos = sinusoid_pos_encod(seq_len=seq_len, dim=dim).unsqueeze(0)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Conv1d(in_channels=384, 
                          out_channels=dim*2,
                          kernel_size=2, padding=0,stride=2)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        query: (B, T_q, D)
        context: (B, T_kv, D)
        context_mask: (B, T_kv) or None
        """
        B, T_q, D = query.size()
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, d)
        KV = self.kv_proj(context.permute(0, 2, 1)).view(B, T_q, 2, self.num_heads*self.head_dim).permute(0, 2, 1, 3)  # (B, 2,T_k, H*d)
        pos_embed = self.pos.to(query.device)  # (1, T_kv, d)
        K,V = torch.unbind(KV, dim=1)  # (B,T_kv, d)
        K = K + pos_embed  # (B,T_kv, d)
        V = V + pos_embed  # (B,T_kv, d)
        K = K.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_kv, d)
        V = V.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_kv, d)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_kv)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, T_q, d)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)  # (B, T_q, D)

        return self.out_proj(out)  

class TranBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3.0, dropout=0.1,seq_len=50):
        super().__init__()

        self.cross_attn = Attention(dim, num_heads, dropout,seq_len)
        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
       
        # Cross-attention
        attn_out = self.cross_attn(x, context)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)

        return x

class CTC(nn.Module):
    
    def __init__(
        self, 
        in_dim =[531,21,23,3],
        hid_dim=64,
        out_dim=14,
        attn_dim=128, # 384
        vocab_size=21, 
        seq_len=100,
        cnn_k=[2,3,4,5,6],
        dropout=0.2,
        conv_k=2,
        max_k= 2,
        depth=6,
        use_attn=True,
        # use_cross=True,
        ):
        super(CTC, self).__init__()

        self.depth = depth
        self.use_attn = use_attn
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=sum(in_dim), out_channels=hid_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_k)
            ) for k in cnn_k
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=len(cnn_k))
        if use_attn:
            self.pos = sinusoid_pos_encod(seq_len=seq_len//2, dim=attn_dim).unsqueeze(0)
            self.maxpool = nn.MaxPool1d(kernel_size=len(cnn_k))
            self.conv_embed = nn.Conv1d(in_channels=hid_dim, out_channels=attn_dim, kernel_size=1,stride = 1,padding=0)

            self.middle = nn.ModuleList([
                TranBlock(attn_dim, 4, mlp_ratio=2.0, dropout=dropout,seq_len=seq_len//2) for _ in range(depth)
            ])

            self.conv_debed = nn.Conv1d(in_channels=attn_dim, out_channels=hid_dim, kernel_size=1,stride= 1,padding=0)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hid_dim, hid_dim, kernel_size=conv_k, padding=conv_k//2),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=k)
            ) for k in [5, 5, 2]
        ])

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.load_weights("./models/s1AMP/CTC/CTC60.pth.tar")

    def load_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=True)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        model_dict = self.state_dict()
        filtered_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_dict and v.shape == model_dict[k].shape
            and not (k.startswith("fc") or k.startswith("middle"))
        }

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        for name, module in self.named_modules():
            if name.startswith("fc") or name.startswith("middle"):
                self.init_weights(module)

        print(f"Loaded {len(filtered_dict)} / {len(model_dict)} parameters (excluding 'fc' and 'middle').")

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        print(f"Initialized weights for {module.__class__.__name__} in {self.__class__.__name__}.")

    def forward(self, AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat, esm_feat):

        x = torch.cat([AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat], dim=2) # BxTxC
        x = x.permute(0, 2, 1) # BxCxT

        out = [conv(x) for conv in self.encoder]
        x = torch.cat(
            [torch.cat([t[..., i:i+1] for t in out], dim=2) for i in range(out[0].shape[-1])],
            dim=2) # BxCx3*T
        
        x = self.maxpool(x) # BxCxT
 
        if self.use_attn:
            x = self.conv_embed(x)
            x = x.permute(0, 2, 1) # BxTxC

            pos_embed = self.pos.to(x.device)  # (1, T_kv, d)
            x = x + pos_embed
            

            i = 0
            for block in self.middle:
                x = block(x, esm_feat[:,(8-self.depth)+i,...])
                i += 1

            x = self.conv_debed(x.permute(0, 2, 1)) # BxCxT

        for layer in self.decoder: 
            x = layer(x) # BxHDx1

        x = x.mean(dim=2) # BxHD
        x = self.drop(x)
        x = self.fc(x)
        return self.sigmoid(x)

if __name__ == '__main__':

    # model = TranCNN()
    # model = SequenceMultiTypeMultiCNN_1()
    # model = CNN_1()
    model = CTC(use_attn=True)
    AAI_embed = torch.randn(2, 100, 531)
    PAAC_embed = torch.randn(2, 100, 3)
    BLOSUM62_embed = torch.randn(2, 100, 23)
    OH_embed = torch.randn(2, 100, 21)
    esm_embed = torch.randn(2,8,100, 384)
    # model.load_weights("./models/s1AMP/CTC/CTC60.pth.tar")
    output = model(AAI_embed, OH_embed, BLOSUM62_embed, PAAC_embed,esm_embed)
    print(output.shape)

