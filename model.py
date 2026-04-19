import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """每个样本的随机路径丢弃（随机深度），应用于残差块主路径时使用。

    与为EfficientNet等网络创建的DropConnect实现相同，但原名称容易混淆，
    因为'Drop Connect'在另一篇论文中是一种不同形式的dropout。
    参考讨论: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    这里改用'drop path'作为层和参数名称，而不是混用DropConnect。

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配不同维度的张量，不仅限于2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """每个样本的随机路径丢弃（随机深度），应用于残差块主路径时使用。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class main_model(nn.Module):
    """
    EDS-Trans: 能量感知双尺度Transformer用于多源无源域适应(MSFDA)。
    实现层次窗口注意力(HWA)，包含全局Swin Transformer分支和局部小窗口自注意力分支，
    通过可学习的alpha系数融合。
    """

    def __init__(self, num_classes, patch_size=4, in_chans=3, embed_dim=96, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, local_window_size=3, qkv_bias=True,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0., norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, HFF_dp=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(96, 192, 384, 768), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()

        ###### 局部分支设置(小窗口自注意力, 论文Eq.3-4) #######
        # 使用浅层Transformer，小窗口(无位移窗口)捕捉细粒度判别特征，避免噪声传播

        self.downsample_layers = nn.ModuleList()   # stem + 3个阶段下采样
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4下采样
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4个特征分辨率阶段
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # 构建每个阶段的Local_block(LW-MSA)堆栈
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # 最终归一化层
        self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        self.conv_head.weight.data.mul_(conv_head_init_scale)
        self.conv_head.bias.data.mul_(conv_head_init_scale)

        ###### 全局分支设置(Swin Transformer, 论文Eq.1-2) ######
        # 使用W-MSA和SW-MSA，更大感受野捕捉全局上下文

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # stage4输出特征矩阵的通道数
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # 将图像分割为不重叠的patch
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        i_layer = 0
        self.layers1 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 1
        self.layers2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 2
        self.layers3 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 3
        self.layers4 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        ###### HWA融合: 可学习的alpha系数 (论文Eq.5) #######
        # F_fu = alpha * F_S + (1 - alpha) * F_L
        # alpha从0开始，按验证精度每个epoch更新
        self.alpha = nn.Parameter(torch.tensor(0.0))

        ###### 层次特征融合块设置 #######

        self.fu1 = HFF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=HFF_dp)
        self.fu2 = HFF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=HFF_dp)
        self.fu3 = HFF_block(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=HFF_dp)
        self.fu4 = HFF_block(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=HFF_dp)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        """
        HWA (层次窗口注意力) 前向传播。
        实现双尺度特征提取和融合，如论文Section 3.2所述。

        全局分支 (Swin Transformer, W-MSA + SW-MSA, Eq.1-2):
          F^l_S  = MLP(LN(W-MSA(LN(z^{l-1})))) + F^{l-1}_S + F^{l-1}_S
          F^{l+1}_S = MLP(LN(SW-MSA(LN(F^l_S)))) + F^l_S + F^l_S

        局部分支 (LW-MSA, 无位移窗口, Eq.3-4):
          F~_L = Z + LW-MSA(LN(Z))
          F_L  = F~_L + MLP(LN(F~_L))

        融合 (Eq.5):
          F_fu = alpha * F_S + (1 - alpha) * F_L
        """
        ###### 全局分支 (Swin Transformer, W-MSA + SW-MSA, 论文Eq.1-2) ######
        x_s, H0, W0 = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H1, W1 = self.layers1(x_s, H0, W0)
        x_s_2, H2, W2 = self.layers2(x_s_1, H1, W1)
        x_s_3, H3, W3 = self.layers3(x_s_2, H2, W2)
        x_s_4, H4, W4 = self.layers4(x_s_3, H3, W3)

        # [B, L, C] --> [B, C, H, W] 使用每个阶段的动态H/W
        B = imgs.shape[0]
        x_s_1 = x_s_1.transpose(1, 2).contiguous().view(B, -1, H1, W1)
        x_s_2 = x_s_2.transpose(1, 2).contiguous().view(B, -1, H2, W2)
        x_s_3 = x_s_3.transpose(1, 2).contiguous().view(B, -1, H3, W3)
        x_s_4 = x_s_4.transpose(1, 2).contiguous().view(B, -1, H4, W4)

        ###### 局部分支 (LW-MSA, 小窗口, 无位移窗口) ######
        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        ###### HWA融合 (Eq.5): F_fu = alpha * F_S + (1-alpha) * F_L ######
        # alpha是可学习标量，通过sigmoid约束到[0,1]
        alpha = torch.sigmoid(self.alpha)
        x_f_1 = alpha * x_s_1 + (1 - alpha) * x_c_1
        x_f_2 = alpha * x_s_2 + (1 - alpha) * x_c_2
        x_f_3 = alpha * x_s_3 + (1 - alpha) * x_c_3
        x_f_4 = alpha * x_s_4 + (1 - alpha) * x_c_4

        ###### HFF聚合路径 ######
        x_f_1 = self.fu1(x_f_1, None)
        x_f_2 = self.fu2(x_f_2, x_f_1)
        x_f_3 = self.fu3(x_f_3, x_f_2)
        x_f_4 = self.fu4(x_f_4, x_f_3)

        x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # 全局平均池化 (N,C,H,W)->(N,C)
        x_fu = self.conv_head(x_fu)

        return x_fu

##### 局部特征块组件 #####

class LayerNorm(nn.Module):
    r""" 支持两种数据格式的LayerNorm: channels_last (默认) 或 channels_first。
    输入的维度顺序。channels_last对应形状为(batch_size, height, width, channels)的输入，
    而channels_first对应形状为(batch_size, channels, height, width)的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Local_block(nn.Module):
    r""" 局部特征块。有两种等价的实现:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; 全部在 (N, C, H, W)
    (2) DwConv -> 排列为 (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; 排列回去
    我们使用(2)因为在PyTorch中速度略快

    参数:
        dim (int): 输入通道数。
        drop_rate (float): 随机深度率。默认: 0.0
    """
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分离卷积
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # 逐点/1x1卷积，用线性层实现
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x

# 层次特征融合块
class HFF_block(nn.Module):
    """
    HFF_block作用于融合的HWA特征(已从全局+局部分支组合)。
    它在每个层次阶段应用空间注意力来细化融合特征，
    并通过残差路径聚合跨阶段上下文。

    当f(前一阶段输出)可用时，它被上采样并连接。
    """
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int, eps=1e-6, data_format="channels_first")
        self.W2 = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 1, bn=True, relu=False)
        self.gelu = nn.GELU()
        self.residual = IRMLP(ch_1 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, f):
        """
        参数:
            l: 当前阶段的HWA融合特征 [B, C, H, W]
            f: 前一阶段输出 (stage-1为None) [B, C//2, H*2, W*2] 或 None
        """
        # 投影当前阶段融合特征
        W_local = self.W_l(l)
        W_local = self.norm2(W_local)
        W_local = self.gelu(W_local)

        # 跨阶段上下文集成
        if f is not None:
            # 上采样前一阶段输出并下采样以匹配当前分辨率
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local], 1)
            X_f = self.norm1(X_f)
            X_f = self.W2(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = self.W(W_local)
            X_f = self.gelu(X_f)

        # 原始融合特征上的空间注意力
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l_att = self.spatial(result)
        l_att = self.sigmoid(l_att) * l_jump

        # 聚合: 融合空间注意特征与跨阶段上下文
        fuse = torch.cat([l_att, X_f], 1)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#### 倒置残差MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

####### 位移窗口MSA #############

class WindowAttention(nn.Module):
    r""" 基于窗口的多头自注意力(W-MSA)模块，带相对位置偏置。
    支持位移和非位移窗口。

    参数:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的高度和宽度。
        num_heads (int): 注意力头数。
        qkv_bias (bool, 可选):  如果为True，为query、key、value添加可学习偏置。默认: True
        attn_drop (float, 可选): 注意力权重的Dropout比率。默认: 0.0
        proj_drop (float, 可选): 输出的Dropout比率。默认: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # 获取窗口内每个token的两两相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 偏移使坐标从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # 拆分为q、k、v（torchscript不能将tensor作为tuple使用）

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: 矩阵乘法 -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # 窗口数量
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: 矩阵乘法 -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

### 全局特征块
class Global_block(nn.Module):
    r""" 来自修改的Swin Transformer块的全局特征块。

    参数:
        dim (int): 输入通道数。
        num_heads (int): 注意力头数。
        window_size (int): 窗口大小。
        shift_size (int): SW-MSA的位移大小。
        qkv_bias (bool, 可选): 如果为True，为query、key、value添加可学习偏置。默认: True
        drop (float, 可选): Dropout率。默认: 0.0
        attn_drop (float, 可选): 注意力Dropout率。默认: 0.0
        drop_path (float, 可选): 随机深度率。默认: 0.0
        act_layer (nn.Module, 可选): 激活层。默认: nn.GELU
        norm_layer (nn.Module, 可选): 归一化层。默认: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = act_layer()

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环位移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA 注意力计算
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # 逆循环位移
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:

            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = shortcut + self.drop_path(x)

        return x

class BasicLayer(nn.Module):
    """
    一个阶段的下采样和全局特征块。

    参数:
        dim (int): 输入通道数。
        depth (int): 块数。
        num_heads (int): 注意力头数。
        window_size (int): 局部窗口大小。
        mlp_ratio (float): mlp隐层维度与嵌入维度的比率。
        qkv_bias (bool, 可选): 如果为True，为query、key、value添加可学习偏置。默认: True
        drop (float, 可选): Dropout率。默认: 0.0
        attn_drop (float, 可选): 注意力Dropout率。默认: 0.0
        drop_path (float | tuple[float], 可选): 随机深度率。默认: 0.0
        norm_layer (nn.Module, 可选): 归一化层。默认: nn.LayerNorm
        downsample (nn.Module | None, 可选): 层末尾的下采样层。默认: None
        use_checkpoint (bool): 是否使用checkpointing来节省内存。默认: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # 构建全局特征块列表
        self.blocks = nn.ModuleList([
            Global_block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch合并层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # 计算SW-MSA的注意力掩码
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = self.downsample(x, H, W)         # patch合并: stage2输入[6,3136,96] 输出[6,784,192]
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:                  # 全局特征块前向传播
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W

def window_partition(x, window_size: int):
    """
    参数:
        x: (B, H, W, C)
        window_size (int): 窗口大小(M)

    返回:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    参数:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小(M)
        H (int): 图像高度
        W (int): 图像宽度

    返回:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """
    2D图像到Patch嵌入
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # 填充
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # 填充最后3个维度
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    r""" Patch合并层。

    参数:
        dim (int): 输入通道数。
        norm_layer (nn.Module, 可选): 归一化层。默认: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim//2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征大小错误"

        x = x.view(B, H, W, C)

        # 填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 填充最后3个维度，从最后一个维度开始向前。
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

def HiFuse_Tiny(num_classes: int):
    model = main_model(depths=(2, 2, 2, 2),
                     conv_depths=(2, 2, 2, 2),
                     num_classes=num_classes)
    return model


def HiFuse_Small(num_classes: int):
    model = main_model(depths=(2, 2, 6, 2),
                     conv_depths=(2, 2, 6, 2),
                     num_classes=num_classes)
    return model

def HiFuse_Base(num_classes: int):
    model = main_model(depths=(2, 2, 18, 2),
                     conv_depths=(2, 2, 18, 2),
                     num_classes=num_classes)
    return model

class LocalFeatureWithAttention(nn.Module):
    def __init__(self, ch, r_2, ch_int, drop_rate=0.):
        super(LocalFeatureWithAttention, self).__init__()
        self.local_block = Local_block(ch, drop_rate)
        self.spatial_attention = HFF_block(ch, ch, r_2, ch_int, ch, drop_rate)

    def forward(self, x):
        x = self.local_block(x)
        x = self.spatial_attention(x)
        return x