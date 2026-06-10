import torch
import torch.nn as nn
import random
import os

seed = 512
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
# np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
# torch.cuda.set_device(6)
device = torch.device('cuda')
pin_memory = True
prefetch_factor = 4

# 实现一个排列不变网络的Actor，用于生成动作。


class GraphAttention(nn.Module):
    """
    单头图注意力层 (GAT)
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        # h: [B, N, F_in]
        Wh = self.W(h)  # [B, N, F_out]
        B, N, F = Wh.size()

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F]
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, F]
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)  # [B, N, N]

        attention = torch.softmax(e, dim=2)  # [B, N, N]
        h_prime = torch.bmm(attention, Wh)   # [B, N, F]
        return h_prime

class ChannelAttention(nn.Module):
    """
    通道注意力（Squeeze-and-Excitation）
    """
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C]
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)  # [B, C]
        w = self.fc(y)                                  # [B, C]
        return x * w

class PermInvariantQNNActor(nn.Module):
    def __init__(self, in_invar_dim, non_invar_dim,
                 out_dim, s_dim=4, a_dim=3,
                 block_size=1, num_moments=1):
        super(PermInvariantQNNActor, self).__init__()

        # 基本参数
        self.s_dim      = s_dim
        self.block_size = block_size
        self.num_blocks = s_dim // block_size
        self.non_invar  = non_invar_dim
        self.num_mom    = num_moments
        self.out_dim    = out_dim

        # 1) 图注意力层
        self.gat = GraphAttention(
            in_features=block_size,
            out_features=block_size
        ).to(device)

        # 2) 通道注意力
        self.ca = ChannelAttention(channels=self.s_dim, reduction=4).to(device)

        # 3) 后续 moment encoder（与原结构一致）
        self.moment_encoder_net = nn.Sequential(
            nn.Linear(self.s_dim, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.num_mom),
            nn.LeakyReLU(),
            nn.Linear(self.num_mom, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Softmax(dim=-1),
            nn.Linear(20, self.out_dim)
        ).to(device)

        # 4) decoder_net（融合非置换特征）
        self.decoder_net = nn.Sequential(
            nn.Linear(self.out_dim + self.non_invar, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.out_dim)
        ).to(device)

    def forward(self, invar_input, non_invar_input=None):
        # 1) reshape -> [B, N, F]
        x = invar_input.view(-1, self.num_blocks, self.block_size).to(device)
        # 2) GAT 聚合
        x = self.gat(x)                                # [B, N, F]
        # 3) flatten -> [B, s_dim]
        x = x.view(-1, self.s_dim)
        # 4) 通道加权
        x = self.ca(x)                                 # [B, s_dim]
        # 5) moment encoder -> [B, out_dim]
        x = self.moment_encoder_net(x)

        # 6) 融合非置换特征并解码
        if non_invar_input is not None:
            x = torch.cat([x, non_invar_input.to(device)], dim=-1)
            x = self.decoder_net(x)
        return x

class PermInvariantQNNCritic(nn.Module):
    def __init__(self, in_invar_dim, non_invar_dim,
                 out_dim, s_dim=4, a_dim=3,
                 block_size=1, num_moments=1):
        super(PermInvariantQNNCritic, self).__init__()

        # 基本参数
        self.s_dim      = s_dim
        self.block_size = block_size
        self.num_blocks = s_dim // block_size
        self.non_invar  = non_invar_dim
        self.num_mom    = num_moments
        self.out_dim    = out_dim

        # 1) 图注意力
        self.gat = GraphAttention(
            in_features=block_size,
            out_features=block_size
        ).to(device)

        # 2) 通道注意力
        self.ca = ChannelAttention(channels=self.s_dim, reduction=4).to(device)

        # 3) moment encoder
        self.moment_encoder_net = nn.Sequential(
            nn.Linear(self.s_dim, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.num_mom),
            nn.LeakyReLU(),
            nn.Linear(self.num_mom, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.out_dim)
        ).to(device)

        # 4) decoder_net
        self.decoder_net = nn.Sequential(
            nn.Linear(self.out_dim + self.non_invar, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.out_dim)
        ).to(device)

    def forward(self, state, action, non_invar_input=None):
        # 张量 & reshape
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32)
        s = state.contiguous().view(-1, self.num_blocks, self.block_size).to(device)
        a = action.contiguous().view(-1, self.num_blocks, self.block_size).to(device)
        x = torch.cat([s, a], dim=0)  # [2B, N, F]

        # 1) GAT
        x = self.gat(x)               # [2B, N, F]
        # 2) flatten & CA
        x = x.view(-1, self.s_dim)    # [2B, s_dim]
        x = self.ca(x)                # [2B, s_dim]
        # 3) moment encoder
        x = self.moment_encoder_net(x)  # [2B, out_dim]

        # 4) 只 decode 前 B 个样本
        if non_invar_input is not None:
            x0 = x[:state.size(0)]
            x0 = torch.cat([x0, non_invar_input.to(device)], dim=-1)
            x0 = self.decoder_net(x0)
            return x0
        return x
