import torch
import torch.nn as nn
import math
from functools import partial

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


class mMHSA(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=4):
        super(mMHSA, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.p_linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.p_linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.p_linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self.c_linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.c_linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.c_linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in, "Input dimension does not match"

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        odd_indices = torch.arange(nh) % 2 == 1

        Pq = self.p_linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        Pk = self.p_linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        Pv = self.p_linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        Cq = self.c_linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        Ck = self.c_linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        Cv = self.c_linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        att1 = torch.matmul(Pq, Pk.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        att1 = torch.softmax(att1, dim=-1)  # batch, nh, n, n

        att2 = torch.matmul(Cq, Ck.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        att2 = torch.softmax(att2, dim=-1) 

        att1[:, odd_indices, :, :], att2[:, odd_indices, :, :] = att2[:, odd_indices, :, :], att1[:, odd_indices, :, :]

        v1 = torch.matmul(att1, Pv)  # batch, nh, n, dv
        v1 = v1.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        v2 = torch.matmul(att2,Cv)  # batch, nh, n, dv
        v2 = v2.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        return v1, v2

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=24, hidden_dim=512, dropout=0.3):
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

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
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
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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
        self.attn = mMHSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Embed(nn.Module):
    """Sequence Embedding to Transformer Embedding"""

    def __init__(self, in_chans, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
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

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
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
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class SequenceMultiTypeMultiCNN_1(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, d_input=[531,21,23,3],
                vocab_size=None, seq_len=None,
                dropout=0.1,d_another_h=64,k_cnn=[2,3,4,5,6],d_output=1):
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
        return x

class SequenceTransformer(nn.Module):
    """AMP Sequence Transformer"""

    def __init__(
        self,
        in_chans=[531, 21, 23, 3],
        num_classes=0,
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
        self.Cpatch = Embed(in_chans[1]+in_chans[2], embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
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
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


if __name__ == '__main__':
    # m = mMHSA(64, 64, 64)
    # x = torch.randn(2, 16, 64)
    # y1, y2 = m(x)
    # print(y1.shape, y2.shape)
    model = SequenceMultiTypeMultiCNN_1(d_input=[531, 21, 23, 3], vocab_size=21, seq_len=200, dropout=0.1, d_another_h=128, k_cnn=[2, 3, 4, 5, 6], d_output=1)
    onehot_feat = torch.randn(2, 200, 21)
    AAI_feat = torch.randn(2, 200, 531)
    BLOSUM62_feat = torch.randn(2, 200, 23)
    PAAC_feat = torch.randn(2, 200, 3)
    PC6_feat = torch.randn(2, 200, 3)
    output = model(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
    print(output.shape)
