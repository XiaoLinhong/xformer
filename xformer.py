## 特色: 
### 1. 基于风向、风速和城市距离动态计算空间权重:  风场感知 的邻居动态权重机制，比传统全局空间注意力有效，能过滤掉远距离、逆风方向上的无效城市影响
### 2. 直接面向城市级别预测, 避免了传统方法里大规模站点数据（动辄成百上千个站点）的稀疏、动态站点问题, 城市粒度 预测，更符合业务实际部署需求
### 3. 双流模型编码（气象-污染分开编码）: 气象和污染特征 分流编码 避免了特征意义混乱问题，提升模型泛化能力。
### 4. 编码预报时长：随着预报步长增长，气象主导作用增强，污染初值影响衰减。
### 5. 不同城市采用不同的权重矩阵，考虑城市间的差异性。
### 6. 损失函数设计：采用分级损失函数，强化重污染等级的预测精度。时间衰减，越老的数据权重越小。
### 7. 对数化处理：采用对数化处理，避免了污染物浓度为负值的情况，更加符合正态分布
### 8. 由于人为排放清单导致的年际变化，会影响模型对未来的预报，去掉年变化趋势

import os
import time
import pickle
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

def haversine_distance_matrix(coords):
    """
    coords: [N, 2] -> (lat, lon) in degrees
    return: [N, N] -> distance matrix in km
    """
    R = 6371.0  # 地球半径，单位：km
    lat = torch.deg2rad(coords[:, 0]).unsqueeze(1)  # [N, 1]
    lon = torch.deg2rad(coords[:, 1]).unsqueeze(1)  # [N, 1]
    dlat = lat - lat.T  # [N, N]
    dlon = lon - lon.T  # [N, N]
    a = torch.sin(dlat / 2)**2 + torch.cos(lat) * torch.cos(lat.T) * torch.sin(dlon / 2)**2
    d = R * 2 * torch.arcsin(torch.sqrt(a))
    return d  # [N, N]

def geo_ngb_city_idx_by_batch(geo_infos, wind_spd, wind_dir, k_neighbors=4, radius=500, batch_size=512):
    ''' T如果很大, 计算影响城市时，内存使用会爆炸 '''
    T, _ = wind_spd.shape

    # 初始化列表用于存储分批的结果
    all_ngb_idx, all_ngb_wgt, all_ngb_arc, all_ngb_dst = [], [], [], []

    # 按批次处理 wind_spd 和 wind_dir
    for t_batch in range(0, T, batch_size):
        t_end = min(t_batch + batch_size, T)  # 当前批次的结束索引
        # 获取当前批次的风速和风向
        wind_spd_batch = wind_spd[t_batch:t_end]  # [batch_size, S]
        wind_dir_batch = wind_dir[t_batch:t_end]  # [batch_size, S]
        # 调用 geo_neighbor_city_idx 函数计算每个批次的邻居城市信息
        ngb_idx, ngb_wgt, ngb_arc, ngb_dst = geo_neighbor_city_idx(geo_infos, wind_spd_batch, wind_dir_batch, k_neighbors, radius)
        # 将当前批次的结果添加到列表中
        all_ngb_idx.append(ngb_idx)
        all_ngb_wgt.append(ngb_wgt)
        all_ngb_arc.append(ngb_arc)
        all_ngb_dst.append(ngb_dst)
    # 将所有批次的结果拼接在一起
    ngb_idx = torch.cat(all_ngb_idx, dim=0)  # [T, S, k_neighbors]
    ngb_wgt = torch.cat(all_ngb_wgt, dim=0)  # [T, S, k_neighbors]
    ngb_arc = torch.cat(all_ngb_arc, dim=0)  # [T, S, k_neighbors]
    ngb_dst = torch.cat(all_ngb_dst, dim=0)  # [T, S, k_neighbors]

    return ngb_idx, ngb_wgt, ngb_arc, ngb_dst

def geo_neighbor_city_idx(geo_infos, wind_spd, wind_dir, k_neighbors=4, radius=500):
    """ 根据风向, 获取对每个城市最有影响的邻居城市 """
    T, S = wind_spd.shape
    city_coords, dist_matrix = geo_infos['coords'], geo_infos['dist_matrix']  # dist_matrix [S, S]
    
    # 计算单位风向向量
    wind_rad = wind_dir * (torch.pi / 180)  # 转弧度，风的来向
    wind_vector = - torch.stack([torch.sin(wind_rad), torch.cos(wind_rad)], dim=-1)  # [T, S, 2]

    # 目标城市指向其他城市的向量:  [1, S, 2] - [S, 1, 2] => [S, S, 2]
    city_coords = city_coords[:, [1, 0]]  # [latitude, longitude] => [longitude, latitude]
    location_vector = F.normalize(city_coords.unsqueeze(0) - city_coords.unsqueeze(1), dim=-1)

    # 风向和城市向量夹角的cos值
    alignment = (wind_vector.unsqueeze(2) * location_vector.unsqueeze(0)).sum(dim=-1)  # [T, S, S]

    # 城市自身与自己的夹角为最大(1)，也就是自己会对自己有影响
    alignment = alignment.masked_fill(torch.eye(city_coords.shape[0], device=alignment.device).bool(), 1)

    # 每个城市的影响半径（单位：公里），注意风速不能为负值
    wind_radius = (wind_spd + 1.0) * 3.6 * 10  # [T, S]，风速转化为公里/小时，乘以10表示影响范围

    # 根据风向，修正对城市的影响半径
    wind_weight = (wind_radius.unsqueeze(2) + 0.1) * alignment.clamp(min=0.1)  # [T, S, S]

    # 根据距离和半径计算影响系数
    dist_matrix_ = dist_matrix.unsqueeze(0)  # [S, S] -> [1, S, S]，扩展为 [T, S, S] 格式
    # 影响函数：
    # - d <= r : 影响系数为 1
    # - d > r  : 影响系数为 exp(- (d - r) / r)
    score = torch.where(
        dist_matrix_ <= wind_weight,  # [T, S, S]，影响半径内，系数为1
        torch.ones_like(dist_matrix_),  # 影响半径内，系数为1
        torch.exp(- (dist_matrix_ - wind_weight) / (wind_weight + 1e-6))  # 半径外，指数衰减
    )  # [T, S, S]

    # 影响目标城市最显著的邻居
    ngb_wgt, ngb_idx = torch.topk(score, k_neighbors, dim=-2)  # [T, k_neighbors, S]
    ngb_wgt = ngb_wgt.permute(0, 2, 1)  # [T, k_neighbors, S] => [T, S, k_neighbors]
    ngb_idx = ngb_idx.permute(0, 2, 1)  # [T, k_neighbors, S] => [T, S, k_neighbors]

    # 选取 ngb_idx 对应的 alignment，扩展成 [T, S, k_neighbors]
    alignment = alignment.permute(0, 2, 1)
    ngb_arc = torch.gather(alignment, dim=-1, index=ngb_idx)  # [T, S, k_neighbors]
    
    # dist_matrix_ 是 [1, S, S]，需要扩展为 [T, S, S]，然后按 ngb_idx 索引
    ngb_dst = torch.gather(dist_matrix_.expand(T, S, S), dim=-1, index=ngb_idx)

    # 返回最终的邻居城市索引，权重，夹角（弧度），以及归一化后的距离（单位：km）
    return ngb_idx, ngb_wgt, ngb_arc, ngb_dst / radius  # ngb_dst / radius 归一化距离 [T, S, k_neighbors]

def build_rope(seq_len, dim, device, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    position = torch.arange(seq_len, device=device).float()
    freq = torch.einsum('i,j->ij', position, theta)
    rope = torch.cat([torch.cos(freq), torch.sin(freq)], dim=-1)
    return rope

def apply_rope(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = rope[..., ::2], rope[..., 1::2]
    x_rope = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rope.flatten(-2)

# ====== 时间维注意力 ======
class TemporalAttention(nn.Module):
    def __init__(self, C, num_heads=2, dropout=0., max_seq_len=24):
        super().__init__()
        assert C % num_heads == 0
        self.num_heads = num_heads
        self.d = C // num_heads
        self.qkv = nn.Linear(C, C * 3, bias=False)
        self.proj = nn.Linear(C, C)
        self.proj_drop = nn.Dropout(dropout)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, max_seq_len, C))
        # self.register_buffer('rope', build_rope(max_seq_len, self.d, device='cpu')) # 预先计算

    def forward(self, x):
        B, S, T, C = x.shape  # [B, S, T, C]
        x = x + self.pos_embedding[:, :, :T, :]  # [B, S, T, C] + [1, 1, T, C]
        qkv = self.qkv(x).reshape(B, S, T, 3, self.num_heads, self.d).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, S, heads, T, d]

        # rope = self.rope[:T, :].to(x.device, x.dtype)  # 按需slice
        # q = apply_rope(q, rope)
        # k = apply_rope(k, rope)

        attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)  # [B, S, heads, T, T]

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(2, 3).reshape(B, S, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ====== 空间维注意力 ======
class SpatialAttention(nn.Module):
    def __init__(self, C, num_heads=2, dropout=0.):
        super().__init__()
        assert C % num_heads == 0
        self.num_heads = num_heads
        self.d = C // num_heads
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C+2, C, bias=False) # 动态加入neighbor的距离和方位
        self.v_proj = nn.Linear(C+2, C, bias=False) # 动态加入neighbor的距离和方位, 相当于位置编码信息

        self.proj = nn.Linear(C, C)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, spatial_info=None):
        B, S, T, C = x.shape  # [B, S, T, C]

        x_ = x.permute(0, 2, 1, 3).reshape(B * T, S, C)  # [B*T, S, C]
        q = self.q_proj(x_).reshape(B * T, S, self.num_heads, self.d).permute(0, 2, 1, 3)  # [B*T, heads, S, d]

        if spatial_info is not None:
            spatial_idx, spatial_wgt, alignment, dist = spatial_info  # [B, T, S, k_neighbors]
            spatial_idx = spatial_idx.reshape(B * T, S, -1)  # [B*T, S, -k_neighbors]
            spatial_wgt = spatial_wgt.reshape(B * T, S, -1)  # [B*T, S, -k_neighbors]
            alignment = alignment.reshape(B * T, S, -1)  # [B*T, S, -k_neighbors]
            dist = dist.reshape(B * T, S, -1)  # [B*T, S, k_neighbors]

            neighbors_x = torch.take_along_dim(
                x_.unsqueeze(1).expand(-1, S, -1, -1),  
                spatial_idx.unsqueeze(-1).expand(-1, -1, -1, C), 
                dim=2
            )
            neighbors_x = torch.cat([neighbors_x, alignment.unsqueeze(-1), dist.unsqueeze(-1)], dim=-1)

            # 能不能在这里, 把距离和方位 融合到 neighbors_x 中
            # [B*T, S, k_neighbors, C+2]  -> [B*T, S, k_neighbors, C] -> B*T, S, k, heads, d] -> B*T, heads, S, k, d]
            k_neighbors = self.k_proj(neighbors_x).reshape(B * T, S, -1, self.num_heads, self.d).permute(0, 3, 1, 2, 4)
            v_neighbors = self.v_proj(neighbors_x).reshape(B * T, S, -1, self.num_heads, self.d).permute(0, 3, 1, 2, 4)

            q_expanded = q.unsqueeze(3)  # [B*T, heads, S, 1, d]

            # 计算注意力得分
            attn = torch.matmul(q_expanded, k_neighbors.transpose(-2, -1)).squeeze(3)  # [B*T, heads, S, k]
            attn = attn / (self.d ** 0.5)
            attn = attn + torch.log(spatial_wgt.unsqueeze(1) + 1e-6)  # [B*T, heads, S, k]

            attn = attn.softmax(dim=-1)  # [B*T, heads, S, k]

            # 计算加权输出
            out = torch.matmul(attn.unsqueeze(-2), v_neighbors).squeeze(-2)  # [B*T, heads, S, d]
            out = out.permute(0, 2, 1, 3).reshape(B * T, S, -1)  # [B*T, S, C]
        else:
            k = self.k_proj(x_).reshape(B * T, S, self.num_heads, self.d).permute(0, 2, 1, 3)  # [B*T, heads, S, d]
            v = self.v_proj(x_).reshape(B * T, S, self.num_heads, self.d).permute(0, 2, 1, 3)  # [B*T, heads, S, d]

            attn = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)  # [B*T, heads, S, S]
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B * T, S, C)  # [B*T, S, C]

        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, T, S, C).permute(0, 2, 1, 3)  # [B, S, T, C]
        return out

# ====== XFormer Block ======
class XFormerBlock(nn.Module):
    def __init__(self, C, num_heads=2, mlp_dim=128, dropout=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(C)
        self.spatial_attn = SpatialAttention(C, num_heads, dropout)

        self.norm2 = nn.LayerNorm(C)
        self.ffn1 = nn.Sequential(
            nn.Linear(C, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, C), nn.Dropout(dropout)
        )

        self.norm3 = nn.LayerNorm(C)
        self.temporal_attn = TemporalAttention(C, num_heads, dropout)

        self.norm4 = nn.LayerNorm(C)
        self.ffn2 = nn.Sequential(
            nn.Linear(C, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, C), nn.Dropout(dropout)
        )

    def forward(self, x, spatial_info=None):
        x = x + self.spatial_attn(self.norm1(x), spatial_info)
        x = x + self.ffn1(self.norm2(x))
        x = x + self.temporal_attn(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))
        return x

# ====== 预报时间感知编码器 ======
class TimeAwareEncoder(nn.Module):
    def __init__(self, poll_dim, mete_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.poll_enc = nn.Linear(poll_dim + 1, hidden_dim // 2)  # 加时间步长特征
        self.mete_enc = nn.Linear(mete_dim, hidden_dim // 2)

    def forward(self, poll_x, mete_x, t_step=0, mete_only=False):
        """
        poll_x: [B, T, S, poll_dim]
        mete_x: [B, T, S, mete_dim]
        return: [B, T, S, hidden_dim]
        """
        B, T, S, _ = mete_x.shape

        mete_feat = self.mete_enc(mete_x)  # [B, T, S, hidden_dim // 2]

        if mete_only:
            # 直接返回气象 + 没有污染信息
            zero_poll = torch.zeros_like(mete_feat)
            return torch.cat([mete_feat, zero_poll], dim=-1)

        # 构造 [B, T, S, 1] tensor, t_step/10. \in [0, 1]
        t_tensor = torch.full((B, T, S, 1), float(t_step/100.), dtype=poll_x.dtype, device=poll_x.device)

        # 拼接预报时间信息和污染物数据
        poll_input = torch.cat([poll_x, t_tensor], dim=-1)  # [B, T, S, poll_dim + 1]

        poll_feat = self.poll_enc(poll_input)               # [B, T, S, hidden_dim//2]

        # 拼接两部分特征： [B, T, S, hidden_dim]
        fused_feat = torch.cat([mete_feat, poll_feat], dim=-1)

        return fused_feat

# ====== XFormer 主模型 ======
class XFormer(nn.Module):
    def __init__(self, poll_dim, mete_dim, C_out, S=334, hidden_dim=32, depth=2, num_heads=2, mlp_dim=64, dropout=0.2):
        super().__init__()
        self.C_out = C_out
        self.poll_dim = poll_dim

        self.enc = TimeAwareEncoder(poll_dim, mete_dim, hidden_dim)
        self.ST_blocks = nn.ModuleList([ XFormerBlock(hidden_dim, num_heads, mlp_dim, dropout) for _ in range(depth) ])

        self.ln = nn.LayerNorm(depth*hidden_dim)
        self.proj = nn.Linear(depth*hidden_dim, hidden_dim)
        # self.proj_drop = nn.Dropout(dropout)

        self.W_dec = nn.Parameter(torch.randn(S, C_out, hidden_dim))  # [S, C_out, hidden_dim]
        self.b_dec = nn.Parameter(torch.randn(S, C_out))              # [S, C_out]

        self.reset_parameters()  # 初始化

    def reset_parameters(self):
        init.xavier_uniform_(self.W_dec)  # Xavier 初始化
        init.zeros_(self.b_dec)           # 偏置初始化为 0

    def forward(self, x, spatial_info=None, t_step=0, mete_only=False):
        # x[B, T, S, C_in]
        B, T, S, _ = x.shape
        poll_x = x[..., :self.poll_dim]
        mete_x = x[..., self.poll_dim:]
        h = self.enc(poll_x, mete_x, t_step=t_step, mete_only=mete_only) # [B, T, S, C_in] => [B, T, S, hidden_dim]
        h = h.permute(0, 2, 1, 3) # [B, T, S, hidden_dim] => [B, S, T, hidden_dim]

        d = []  # deterministic states
        for st in self.ST_blocks:
            h = st(h, spatial_info)
            d.append(h)  # [B, S, T, hidden_dim]

        # 多层作为输出
        # d = torch.stack(d)[:, :, :, -1]  # [depth, B, S, T, hidden_dim] => [depth, B, S, hidden_dim]
        # h = d.permute(1, 2, 0, 3).reshape(B, -1) # [B, S*depth*hidden_dim]
        # h = self.ln(h).reshape(B, S, -1) # [B, S*depth*hidden_dim] => [B, S, depth*hidden_dim]

        d = torch.stack(d)  # [depth, B, S, T, hidden_dim]
        h = d.permute(1, 2, 3, 0, 4).reshape(B, S, T, -1) # [B, S, T, depth*hidden_dim]

        h = h[:, :, -1, :] # 取最后一帧 => [B, S, hidden_dim]，应该融合了时空信号
        h = self.ln(h) # # 正则层
        h = self.proj(h)

        # 使用 einsum 做 batched linear: 对每个城市, 用各自的 W_dec[s]
        y = torch.einsum('bsh,sch->bsc', h, self.W_dec) + self.b_dec  # [B, S, C_out]
        y = y.unsqueeze(1)  # [B, 1, S, C_out]

        return y

    def count_parameters(self, verbose=True):
        total_params = 0
        trainable_params = 0

        if verbose:
            print(f"{'模块':<40} {'总参数量':>12} {'可训练':>12}")
            print("=" * 70)

        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count

            if verbose:
                print(f"{name:<40} {param_count:>12,} {param_count if param.requires_grad else 0:>12,}")

        print("=" * 70)
        print(f"{'总计':<40} {total_params:>12,} {trainable_params:>12,}")

    def save(self, fileName):
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        torch.save(self.state_dict(), fileName)

    def load(self, fileName, map_location=None):
        state_dict = torch.load(fileName, map_location=map_location)
        self.load_state_dict(state_dict)

def get_x_and_y(batch, C_out=6, t_step=0, y_preds=[], min_T=4, max_T=8):
    ''' 根据时间步长，和最小历史时间与最大历史时间的限制，获取训练数据 '''
    # B, P, S, C_in = batch.shape
    beg = 0 if min_T+t_step < max_T else min_T + t_step - max_T
    end = min_T+t_step if min_T+t_step < max_T else beg + max_T
    x = batch[:, beg:end, :, :] # [B, T, S, C_out]
    y = batch[:, end:end+1, :, :C_out] # [B, 1, S, C_out]
    # y_trend = y - batch[:, end-1, :, :C_out]

    # 用预报的污染数据，代替观测污染数据，实现长时间预报
    for i, y_pred in enumerate(reversed(y_preds), start=1):
        x[:, -i, :, :C_out] = y_pred[:, 0, :, :] # 只替换污染物部分
    return x, y, beg, end

def make_dataset(beg_idx, batch_size, P=24, val_ratio=0.1):
    ''' 只做时间索引的扰动，通过索引对dataset进行切片，节约内存 '''
    total_size = beg_idx.shape[0] # 训练数据的总大小
    dataset = TensorDataset(beg_idx)
    # ========= 数据划分 ===========
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    train_idx, val_idx = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_idx, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ========= 训练函数 ===========
def run_epoch(model, dataset, data_loader, C_out=6, min_T=4, max_T=8, P=24, mete_only=False, optimizer=None, poll_hour=6):
    data, rank_wgt, ngb_idx, ngb_wgt, ngb_arc, ngb_dst = dataset
    is_train = optimizer is not None
    total_loss = 0

    count = 0
    sum_sq_err = torch.zeros(C_out).to(data.device)

    model.train() if is_train else model.eval()
    # for batch, ngb_idx, ngb_wgt, ngb_arc, ngb_dst in data_loader:
    end_idx = torch.arange(P).to(data.device).view(1, -1)
    for beg_idx, in data_loader:
        idx = beg_idx.view(-1, 1) + end_idx
        data_ = data[idx].clone()  # 动态修改污染物浓度, shape: [batch_size, P, S, C]
        rank_wgt_ = rank_wgt[idx] # 等级权重，加强重污染等级的权重
        ngb_info = ngb_idx[idx], ngb_wgt[idx], ngb_arc[idx], ngb_dst[idx] # shape: [batch_size, P, S, k_neighbors]
        #y_preds = deque(maxlen=max_T)
        y_preds = deque(maxlen=1) # batch 是公用的，只需要每次一个时次就行
        one_loss = 0.
        niter = P - min_T
        assert niter > 0
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            if is_train: optimizer.zero_grad()
            for t_step in range(niter):
                x_batch, y_batch, beg, end = get_x_and_y(data_, C_out, t_step=t_step, y_preds=y_preds, min_T=min_T, max_T=max_T)
                spatial_info = tuple(ngb[:, beg:end, :, :] for ngb in ngb_info)
                if t_step < poll_hour:
                    y_pred = model(x_batch, spatial_info, t_step=t_step, mete_only=mete_only)
                else:
                    y_pred = model(x_batch, spatial_info, t_step=t_step, mete_only=True)

                y_preds.append(y_pred.detach()) # 预报的数据, 注意要断开 autograd 图
                
                # 基于等级权重的均方误差损失
                rank_wgt_batch = rank_wgt_[:, end:end+1, :, :]
                mse = (y_pred - y_batch) ** 2                # [B, 1, S, C]
                #weighted_mse = rank_wgt_batch * mse          # [B, 1, S, C]
                #loss = weighted_mse.mean()
                loss = mse.mean()

                # loss = criterion(y_pred, y_batch) # nn.MSELoss()

                err = (y_pred.detach() - y_batch.detach()) ** 2  # shape: [B, ..., C_out]
                err = err.reshape(-1, C_out)  # flatten 所有时间步与空间点
                sum_sq_err += err.sum(dim=0)  # 对每个通道加和
                count += err.shape[0]         # 总的样本数

                if is_train: loss.backward()
                one_loss += loss.item()
            if is_train: optimizer.step()
        total_loss += one_loss / niter
    rmse = torch.sqrt(sum_sq_err / count)
    return total_loss / len(data_loader), rmse

def compute_mean_std(data, dims=(0, 1)):
    # dims=(0, 1) 对 时间, 城市维度求均值，保留特征维度
    mean = data.mean(dim=dims, keepdim=True)
    std = data.std(dim=dims, keepdim=True)
    return mean, std

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-6) # 加个小常数防止除以0

def denormalize(data_norm, mean, std):
    return data_norm * (std + 1e-6) + mean

def remove_annual_diurnal_cycle(data):
    """
    从 data 中去除年循环和日循环成分
    参数:
        data: ndarray of shape (N, S, C)，按小时连续排列，整数天起始（0时）
    返回:
        data_: ndarray of shape (N, S, C)，去除周期成分后的残差
    """
    N, S, C = data.shape
    data_ = np.zeros_like(data)
    
    # 构造时间索引 t（单位：小时）
    t = np.arange(N)  # shape: (N,)
    
    # 构造年、日循环频率（单位：rad/小时）
    freq_d = 2 * np.pi / 24            # 日循环：周期为24小时
    freq_y = 2 * np.pi / (365.25 * 24) # 年循环：周期为365.25天
    
    # 构造回归设计矩阵 X：包含两个频率的正弦、余弦项 + 常数项
    X = np.stack([
        np.sin(freq_d * t), np.cos(freq_d * t),   # 日循环
        np.sin(freq_y * t), np.cos(freq_y * t),   # 年循环
        np.ones_like(t)                            # 常数项
    ], axis=1)  # shape: (N, 5)

    for s in range(S):
        for c in range(C):
            y = data[:, s, c]
            # 最小二乘拟合周期项 X beta = Y
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_fit = X @ beta # Y 在 X 平面上的投影
            # 去除周期部分，保留残差，Y到X平面的距离
            data_[:, s, c] = y - y_fit
    return data_

def cal_rank_wgts(data, K=10, epsilon=0.05, time_decay_alpha=1.0):
    """
    计算每个数据点的权重，频率越低，权重越高，最后归一化使得时间维度平均为1。
    并且添加时间衰减，越老的数据，影响越小。
    参数：
    data (np.ndarray): 输入数据，形状为 [N, S, C]
    K (int): 将数据分成的区间数（按分位数划分），默认为100
    epsilon (int): 平滑参数，拉普拉斯平滑

    返回：
    np.ndarray: 权重矩阵，形状为 [N, S, C]
    """
    N, S, C = data.shape

    data_new = remove_annual_diurnal_cycle(data) # 去除年日循环成分，统计的时候，更加有代表性

    data_min = data_new.min(axis=0)  # 形状 [S, C]
    data_max = data_new.max(axis=0)  # 形状 [S, C]

    data_low = np.quantile(data_new, 0.01, axis=0)
    data_upp = np.quantile(data_new, 0.99, axis=0)

    # 2. 创建每个站点和污染物的100个bins
    bins = np.linspace(data_low, data_upp, K - 1, axis=-1)  # 输出为 [S, C, K-1]

    # 3. 拼接 min 和 max 作为头尾
    bins = np.concatenate([
        data_min[..., np.newaxis],  # [S, C, 1]
        bins,                       # [S, C, K-1]
        data_max[..., np.newaxis]   # [S, C, 1]
    ], axis=-1)                     # → [S, C, K+1]，共 K 个区间

    # 2. 初始化频率统计，初始化为0，添加平滑项
    freq = np.zeros((S, C, K)) + epsilon

    # 3. 遍历每个区间统计频率
    for k in range(K):
        in_rank = (data_new >= bins[None, :, :, k]) & (data_new < bins[None, :, :, k + 1])  # [N, S, C]
        freq[:, :, k] += in_rank.sum(axis=0) / N  # [S, C]

    # 4. 计算每个数据点属于哪个区间
    data_ = np.expand_dims(data_new, axis=-1)  # [N, S, C, 1]
    in_rank = (data_ >= bins[None, :, :, :-1]) & (data_ < bins[None, :, :, 1:])  # [N, S, C, K]
    bin_idx = in_rank.argmax(axis=-1)  # [N, S, C]

    # 5. 根据区间索引查找频率并计算权重
    wgt = np.zeros((N, S, C))
    for s in range(S):
        for c in range(C):
            wgt[:, s, c] = 1.0 / freq[s, c, bin_idx[:, s, c]] # 反距离权重 1/freq

    wgt = np.clip(wgt, 0.1, 10.0) # 避免特别小的值

    # 极端值处理，这些样本没有太多价值，在计算损失时，给与较小的权重
    wgt[..., 0][data[..., 0]<=1.0] = 0.01
    wgt[..., 1][data[..., 1]<=1.0] = 0.01
    wgt[..., 2][data[..., 1]<0.05] = 0.01
    wgt[..., 3][data[..., 3]<=1.0] = 0.01
    wgt[..., 4][data[..., 4]<=1.0] = 0.01
    wgt[..., 5][data[..., 5]<=1.0] = 0.01

    wgt[..., 0][data[..., 0]>600] = 0.01 # pm25
    wgt[..., 1][data[..., 1]>900] = 0.01 # pm10
    wgt[..., 2][data[..., 2]>50]  = 0.01 # CO
    wgt[..., 3][data[..., 3]>300] = 0.01 # NO2
    wgt[..., 4][data[..., 4]>200] = 0.01 # SO2
    wgt[..., 5][data[..., 5]>600] = 0.01 # O3

    # 时间衰减, 越老的数据，影响越小
    time_wgt = np.exp(time_decay_alpha * np.linspace(0, 1, N))
    wgt = wgt * time_wgt[:, None, None]

    # 6. 归一化，使时间维度上均值为1
    wgt = wgt / wgt.mean(axis=0, keepdims=True)

    # wgt[..., [2:5]] = 1.0 # CO, NO2, SO2
    return wgt

def get_geo_infos(fileName="./dataset/city.csv", demo=False):
    ''' 获取城市信息 '''
    if demo:
        city_coords = [
            [31.2304, 121.4737],  # 上海
            [32.0603, 118.7969],  # 南京
            [30.2741, 120.1551],  # 杭州
            [31.2989, 120.5853],  # 苏州
            [32.3942, 119.4127],  # 镇江
            [31.4912, 120.3119],  # 无锡
        ]
        static_data = torch.randn(len(city_coords), 2) # 海拔, 人口等数据
        return city_coords, static_data
    city_info = pd.read_csv(fileName)
    # code,cityName,lon,lat,hgt,area,pop,forest,
    city_coords = city_info.to_numpy()[:, [3, 2]]
    return city_coords.astype("float32"), city_info.to_numpy()[:, 3:].astype("float32")

def detrend(data, days_per_year=[364, 365, 366, 365, 365, 365, 366]):
    ''' 由于减排措施的实施，污染数据具有显著的年变化趋势，用最后一年的数据作为基准，进行去趋势化 '''
    cumulative_days = np.cumsum([0] + days_per_year)

    epsilon = 1e-3  # 避免 log(0)
    # (N, S, C_out) => (day, 24, S, C_out)
    data_safe = data.reshape(-1, 24, data.shape[-2], data.shape[-1]) + epsilon

    # 创建新数组
    data_norm = np.zeros_like(data_safe)

    # 获取 最有一年的数据 作为基准 (第6年，对应索引6)
    data_last = data_safe[cumulative_days[-2]:cumulative_days[-1]]  # shape: (day, 24, S, C_out)

    # 将过去几年的方差和均值 与 最后一年对齐 exp( (log(x) - this_mean )*last_sigma/this_sigma + last_mean)
    #log_last = np.log(data_last)
    #mean_last = log_last.mean(axis=(0, ))  # shape: (S, C_out)
    #std_last = log_last.std(axis=(0, ))    # shape: (S, C_out)
    
    mean_last = data_last.mean(axis=(0, ))  # shape: (S, C_out)

    # 遍历每一年
    for year_idx in range(len(days_per_year)):
        start_day = cumulative_days[year_idx]
        end_day = cumulative_days[year_idx + 1]
        data_year = data_safe[start_day:end_day]   # shape: (day, 24, S, C_out)

        #log_data = np.log(data_year)
        #mean_year = log_data.mean(axis=(0, 1))    # shape: (S, C_out)
        #std_year = log_data.std(axis=(0, 1))      # shape: (S, C_out)

        mean_year = data_year.mean(axis=(0, ))  # shape: (S, C_out)

        data_norm[start_day:end_day] = data_year * mean_last/mean_year # 只做平均值的订正

        # 防止除以0
        #std_year_safe = np.where(std_year == 0, 1e-6, std_year)
        # 标准化并反变换
        #norm_log = (log_data - mean_year) * (std_last / std_year_safe) + mean_last
        #data_norm[start_day:end_day] = np.exp(norm_log)
    return data_norm.reshape(-1, data.shape[-2], data.shape[-1])
    
def read_train_data(geo_infos, static_data, fileName="./dataset/city_data.pkl", k_neighbors=4, demo=False):
    ''' 训练数据 '''
    C_out = 6 # 预报污染物
    # PM25,PM10,CO,NO2,SO2,O3,PBLH,RAINNCV,SWDOWN,SLP,U10,V10,T2,RH2,hour_cos,month_cos
    S = static_data.shape[0] # 城市数量
    if demo:
        data = torch.randn(1024*S, 16)
    else:
        data = pickle.load(open(fileName,'rb'))  # DataFrame, shape = [N*S, C_in]
    data = data.to_numpy().reshape(-1, S, data.shape[1])  # 转为 numpy, shape = [N, S, C_in]
    if not demo:
        # data = data[:-366*24] # 2024年作为评估时间段
        # data[..., [2, 4]] = detrend(data[..., [2, 4]], days_per_year=[364, 365, 366, 365, 365, 365])
        # data[..., :C_out] = detrend(data[..., :C_out], days_per_year=[364, 365, 366, 365, 365, 365])
        data[..., [2, 4]] = detrend(data[..., [2, 4]], days_per_year=[364, 365, 366, 365, 365, 365, 366])
        # data = data[(364+365)*24:-366*24]

        # 简单质控
        data[..., 2] = data[..., 2]*1000.  # CO, mg => ug
        data[..., :C_out][data[..., :C_out]<1.0] = 1.0
        data[..., 2] = data[..., 2]/1000.  # CO, ug => mg
        data[..., :C_out][data[..., :C_out]>1000] = 1000.
        # data = data[:128] # 测试

    # data[N, S, 6] # 时间，站点，污染物变量，
    # 统计不同站点，不同污染物出现的频率，然后计算一个权重wgt[N, S, 6], 频率越低的，权重越高，N个权重的平均为1
    rank_wgt = cal_rank_wgts(data[..., :C_out], epsilon=0.05, time_decay_alpha=1.0) # [P, S, C_out]
    rank_wgt = torch.tensor(rank_wgt, dtype=torch.float32)
    if not demo:
        # 尽量满足正态分布
        data[..., :C_out] = np.log(data[..., :C_out]) # 对数化
    # 计算风速, 风向, 替换掉 U10 和 V10
    U10 = data[..., 10]
    V10 = data[..., 11]
    wind_spd = np.sqrt(U10**2 + V10**2)
    wind_dir = (270 - np.degrees(np.arctan2(V10, U10))) % 360 # 气象风向，单位: 度
    data[..., 10] = wind_spd
    data[..., 11] = wind_dir

    wind_spd, wind_dir = torch.tensor(wind_spd, dtype=torch.float32), torch.tensor(wind_dir, dtype=torch.float32)

    # [N, T, S, k_neighbors]
    ngb_info = geo_ngb_city_idx_by_batch(geo_infos, wind_spd, wind_dir, k_neighbors=k_neighbors, radius=500)

    # 拼接: 污染数据 + 气象数据 + 静态数据(城市海拔、面积、人口、土地利用)
    data = torch.cat([torch.tensor(data, dtype=torch.float32), static_data.unsqueeze(0).expand(data.shape[0], -1, -1)], dim=-1)
    print('train data shape:', data.shape) # [61344, 334, 31]
    return [data, rank_wgt, *ngb_info], data.shape[-1], C_out

class CFG:
    kargs = dict(S=334, hidden_dim=64, depth=2, num_heads=4, mlp_dim=128, dropout=0.3)
    k_neighbors = 8
    min_T = 8
    max_T = 8
    mete_only = False
    lead_hour = 24 # 预测的时间步长
    poll_hour = 24 # 历史污染数据作为特征的时效，超过这个时间就用纯气象
    mete_model = "mete_model.pth"
    poll_model = "poll_model.pth"

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=3 python xformer.py
    # CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 python xformer.py # 内存爆炸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参数
    k_neighbors = CFG.k_neighbors
    min_T = CFG.min_T
    max_T = CFG.max_T          # 语言大模型中的上下文宽度
    mete_only = CFG.mete_only  # 只用气象数据

    model_file = CFG.mete_model if mete_only else CFG.poll_model

    lead_hour = 1 if mete_only else CFG.lead_hour  # 预测的时间步长

    P = min_T + lead_hour # 预测的时间步长
    retart_file = "restart.pth" # 上次接最佳模型
    # 训练
    epochs = 20
    batch_size = 128
    lr = 1e-3
    lr_scheduler_step = 5  # 每5个epoch调整一次学习率
    lr_scheduler_gamma = 0.5  # 学习率衰减系数

    varNames = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']

    # ========= 读取静态数据 ===========
    city_coords, static_data = get_geo_infos("./dataset/city.csv") # get_demo_geo_infos
    city_coords = torch.tensor(city_coords, dtype=torch.float32)
    static_data = torch.tensor(static_data, dtype=torch.float32)
    geo_dist_matrix = haversine_distance_matrix(city_coords)
    geo_infos ={'coords': city_coords, 'dist_matrix': geo_dist_matrix}

    # ========= 读取训练数据 ===========
    dataset, C_in, C_out = read_train_data(geo_infos, static_data, fileName="./dataset/city_data.pkl", k_neighbors=k_neighbors)
    x_mean, x_std = compute_mean_std(dataset[0])
    dataset[0] = normalize(dataset[0], x_mean, x_std) # 标准化

    dataset = [x.to(device) for x in dataset] # 所有输入数据移动到
    beg_idx = torch.arange(0, dataset[0].shape[0]-P+1, device=device)
    train_loader, val_loader = make_dataset(beg_idx, batch_size, val_ratio=0.2, P=P)

    # ========= 初始化模型 ===========
    model = XFormer(C_out, C_in-C_out, C_out, **CFG.kargs).to(device) # 污染变量, 气象变量, 污染变量
    # model = torch.compile(model)
    if os.path.exists(retart_file):
        print("load: ", retart_file)
        checkpoint = torch.load(retart_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.count_parameters()

    # ========= 损失 + 优化器 ===========
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ========= 学习率调度器 ===========
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # ========= 训练循环 ===========
    best_val_loss = float('inf')
    kargs = dict(C_out=C_out, min_T=min_T, max_T=max_T, P=P, mete_only=mete_only, poll_hour=CFG.poll_hour)
    for epoch in range(epochs):
        start_time = time.time() # 记录开始时间

        # 训练
        train_loss, train_rmse = run_epoch(model, dataset, train_loader, optimizer=optimizer, **kargs)

        # 验证
        val_loss, val_rmse = run_epoch(model, dataset, val_loader, **kargs)

        # 学习率调整
        #scheduler.step()
        scheduler.step(val_loss)

        # 记录结束时间
        end_time = time.time()
        epoch_time = end_time - start_time  # 计算用时（秒）

        # 输出每个epoch的训练和验证损失
        print(f"Epoch {epoch+1:3}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f} sec")
        print( '==> train: ' +' | '.join([f"{name}: {rmse.item()**2:.4f}" for name, rmse in zip(varNames, train_rmse)]))
        print( '==>   val: ' +' | '.join([f"{name}: {rmse.item()**2:.4f}" for name, rmse in zip(varNames, val_rmse)]))

        # 模型保存(保存当前最好的模型)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'x_mean': x_mean,
                'x_std': x_std,
            }, model_file)

    print("Training Complete!")
