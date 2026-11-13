"""
第八周上机任务 —— 任务 2：带注意力的 Seq2Seq（GRU 编码器-解码器）

模型结构（与课件第 19、20、25 页对应）：
    编码器：单向 GRU，依次读入 x_1, ..., x_L，得到每个时刻的隐状态 h_1, ..., h_L。
    解码器：在第 t 步（t = 1, ..., T），
        (1) 计算上下文向量 c_t
            - attn_type = 'none'（无注意力基线，课件第 19 页“改进方式”）：
                  c_t = h_L，所有解码步共用同一个固定的上下文向量；
            - attn_type = 'dot' / 'scaled_dot' / 'additive'（注意力，课件第 20、25 页）：
                  查询 q = s_{t-1}（解码器上一步的隐状态），键 = 值 = [h_1, ..., h_L]，
                  α_{t,j} = softmax_j( s(h_j, s_{t-1}) )，  c_t = Σ_j α_{t,j} h_j；
        (2) 更新隐状态：s_t = GRUCell([e(y_{t-1}); c_t], s_{t-1})，其中 e(·) 为字符嵌入，初始 s_0 = h_L；
        (3) 输出：logits_t = W_o [s_t; c_t] + b_o。

    注意：attn_type 为 'none'、'dot'、'scaled_dot' 时三个模型的参数完全相同，
    唯一的区别在于上下文向量 c_t 是否随解码步 t 变化。
"""
import torch
import torch.nn as nn

from attention import AdditiveScore, attention, dot_score, scaled_dot_score
from utils import SOS_IDX

ATTN_TYPES = ['none', 'dot', 'scaled_dot', 'additive']


class Encoder(nn.Module):
    """单向 GRU 编码器（已实现，无需修改）"""

    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        """
        src: [B, L]
        返回:
            enc_outputs: [B, L, H]，即 h_1, ..., h_L（作为注意力的键和值）
            enc_last:    [B, H]，即最后时刻的隐状态 h_L
        """
        enc_outputs, h_n = self.gru(self.embedding(src))
        return enc_outputs, h_n.squeeze(0)


class Decoder(nn.Module):
    """GRU 解码器（__init__ 已实现，需要完成 forward_step）"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, attn_type='scaled_dot'):
        super().__init__()
        assert attn_type in ATTN_TYPES, f'attn_type 必须是 {ATTN_TYPES} 之一'
        self.attn_type = attn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru_cell = nn.GRUCell(emb_dim + hidden_dim, hidden_dim)  # 输入为 [e(y_{t-1}); c_t]
        self.out = nn.Linear(2 * hidden_dim, vocab_size)               # 输入为 [s_t; c_t]

        # 打分函数：调用方式统一为 self.score_fn(query, keys)
        if attn_type == 'dot':
            self.score_fn = dot_score
        elif attn_type == 'scaled_dot':
            self.score_fn = scaled_dot_score
        elif attn_type == 'additive':
            self.score_fn = AdditiveScore(key_dim=hidden_dim, query_dim=hidden_dim, hidden_dim=hidden_dim)
        else:  # 'none'
            self.score_fn = None

    def forward_step(self, y_prev, s_prev, enc_outputs, enc_last, hard=False):
        """
        解码器的一步（第 t 步）。

        参数:
            y_prev:      [B]，上一步的输出字符 y_{t-1}（第 1 步为 <SOS>）
            s_prev:      [B, H]，解码器上一步的隐状态 s_{t-1}
            enc_outputs: [B, L, H]，编码器所有时刻的隐状态 h_1, ..., h_L
            enc_last:    [B, H]，编码器最后时刻的隐状态 h_L
            hard:        是否使用硬注意力（默认 False；仅在任务 3 使用 --hard_train 时为 True）
        返回:
            logits:  [B, V]，第 t 步各字符的得分（未经过 softmax）
            s_t:     [B, H]，解码器当前步的隐状态
            weights: [B, L]，第 t 步的注意力权重 α_{t,1..L}；attn_type='none' 时返回 None
        """
        # TODO 7: 完成解码器的一步
        #   (1) 计算上下文向量 c_t 和注意力权重 weights：
        #         - self.attn_type == 'none'：c_t = enc_last，weights = None
        #         - 否则：以 s_prev 为查询，enc_outputs 同时作为键和值，
        #                 调用任务 1 中实现的 attention(...)，打分函数为 self.score_fn，并传入 hard 参数
        #   (2) s_t = self.gru_cell([e(y_{t-1}); c_t], s_prev)，其中 e(y_{t-1}) = self.embedding(y_prev)
        #   (3) logits = self.out([s_t; c_t])
        #   提示：拼接使用 torch.cat(..., dim=-1)
        if self.attn_type == 'none':
            context = enc_last
            weights = None
        else:
            context, weights = attention(s_prev, enc_outputs, enc_outputs, self.score_fn, hard=hard)

        embedded = self.embedding(y_prev)
        s_t = self.gru_cell(torch.cat([embedded, context], dim=-1), s_prev)
        logits = self.out(torch.cat([s_t, context], dim=-1))
        return logits, s_t, weights


class Seq2Seq(nn.Module):
    """编码器-解码器整体（已实现，无需修改）"""

    def __init__(self, vocab_size, emb_dim=32, hidden_dim=128, attn_type='scaled_dot'):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, attn_type)

    def forward(self, src, tgt=None, max_len=None, hard=False):
        """
        训练时传入 tgt，使用 teacher forcing：第 t 步的输入是真实的 y_{t-1}；
        测试时 tgt=None，使用贪心解码：第 t 步的输入是模型上一步预测的字符，共解码 max_len 步。
        hard=True 时每一步都使用硬注意力（任务 3）。

        返回:
            logits: [B, T, V]
            attn:   [B, T, L]，attn[b, t, j] 为第 t 个输出对第 j 个输入的注意力权重；无注意力时为 None
        """
        enc_outputs, enc_last = self.encoder(src)
        batch_size = src.size(0)
        num_steps = tgt.size(1) if tgt is not None else max_len

        y_prev = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=src.device)
        s = enc_last  # s_0 = h_L
        all_logits, all_weights = [], []
        for t in range(num_steps):
            logits, s, weights = self.decoder.forward_step(y_prev, s, enc_outputs, enc_last, hard=hard)
            all_logits.append(logits)
            if weights is not None:
                all_weights.append(weights)
            y_prev = tgt[:, t] if tgt is not None else logits.argmax(dim=-1)

        logits = torch.stack(all_logits, dim=1)
        attn = torch.stack(all_weights, dim=1) if all_weights else None
        return logits, attn
