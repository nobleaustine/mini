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


if __name__ == '__main__':

    model = SequenceTransformer()
    
    AAI_embed = torch.randn(2, 200, 531)
    PAAC_embed = torch.randn(2, 200, 3)
    BLOSUM62_embed = torch.randn(2, 200, 23)
    OH_embed = torch.randn(2, 200, 21)

    x = [AAI_embed, PAAC_embed, BLOSUM62_embed, OH_embed]
    output = model(x)
    print(output.shape)
