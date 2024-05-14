import torch

class Quantizer:
    def __init__(self, bits=4):
        # W 的每一行都采用不同的 scale, bits 表示量化的位数
        self.bits = bits
        self.num_grids = 1 << self.bits - 1  # 2 ** self.bits - 1
    
    def get_quantizer_params(self, W):
        self.scalers = torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
        W_max, W_min = W.max(dim=1)[0], W.min(dim=1)[0]
        # 为简单起见, 总假定 W 的每一行都有正有负, 量化采用"非对称量化":
        # 即最大值量化后为 num_grids, 最小值量化后为 0
        assert (torch.all(W_max > 0) and torch.all(W_min < 0)).item()
        self.scalers = (W_max - W_min) / self.num_grids
        # self.zeros = torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
        self.W_max, self.W_min = W_max, W_min
    
    def quantize(self, W):
        # W: (d_in, k)
        quant_int_W = (W - self.W_min[:, None]) / self.scalers[:, None]
        quant_int_W = torch.clamp(torch.round(quant_int_W).long(), 0, self.num_grids)
        quant_float_W = quant_int_W.float() * self.scalers[:, None] + self.W_min[:, None]
        return quant_int_W, quant_float_W

# ==========================================================================
# 不加Lazy Batch-Updates以及Cholesky Reformulation, 并且不断缩小 H_inv 形状的版本
def submatrix(A, i, j):
    A1 = torch.concat([A[:i, :], A[i+1:, :]], axis=0)
    A1 = torch.concat([A1[:, :j], A1[:, j+1:]], axis=1)
    return A1

def true_update_hessian(H, i):
    h = H - 1 / H[i,i] * (H[i].unsqueeze(1) @ H[i].unsqueeze(0))
    h = submatrix(h, i, i)
    return h


def gptq_1(X, W, N, d_in, d_out, quantizer):
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    H = 2 / N * (X @ X.T)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    for i in range(d_in):
        quant_int_w, quant_float_w = quantizer.quantize(W[:, i].unsqueeze(1))  # 量化第 i 列
        Q[:, i] = quant_float_w.squeeze(1)
        # print(W[:, i] - Q[:, i])
        # delta: shape: (d_out, d_in - i)
        delta = - ((W[:, i] - Q[:, i]) / H_inv[0, 0]).unsqueeze(1) @ (H_inv[:, 0].unsqueeze(0))
        Losses[:, i] = (W[:, i] - Q[:, i])**2 / H_inv[0, 0] / 2
        W[:, i:] = W[:, i:] + delta
        # print(H_inv[0, 0], Losses[:, i])
        H_inv = true_update_hessian(H_inv, 0)
    return Q, Losses

# ==============================================================================
# 不加Lazy Batch-Updates以及Cholesky Reformulation, H_inv大小保持不变
def update_hessian(H, i):
    h = H - 1 / H[i,i] * (H[i].unsqueeze(1) @ H[i].unsqueeze(0))
    return h

def gptq_2(X, W, N, d_in, d_out, quantizer):
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    H = 2 / N * (X @ X.T)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    tracks = torch.zeros_like(H_inv)  #
    for i in range(d_in):
        quant_int_w, quant_float_w = quantizer.quantize(W[:, i].unsqueeze(1))  # 量化第 i 列
        Q[:, i] = quant_float_w.squeeze(1)
        # delta: shape: (d_out, d_in)
        delta = - ((W[:, i] - Q[:, i]) / H_inv[i, i]).unsqueeze(1) @ (H_inv[:, i].unsqueeze(0))
        Losses[:, i] = (W[:, i] - Q[:, i])**2 / H_inv[i, i] / 2
        W += delta
        # print(W[:, :i])
        tracks[i:, i] = H_inv[i:, i] / torch.sqrt(H_inv[i, i])
        H_inv = update_hessian(H_inv, i)
    return Q, Losses, tracks


# =======================================================================================
# 包含 Lazy Batch-Updates 以及 Cholesky Reformulation 的完整版 GPTQ 实现
def gptq_3(X, W, N, d_in, d_out, quantizer):
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    H = 2 / N * (X @ X.T)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    block_size = 4  # 假定d_in总能被block_size整除
    L = torch.linalg.cholesky(H_inv, upper=True)
    for i in range(0, d_in, block_size):
        E = torch.zeros((d_out, block_size), dtype=Losses.dtype, device=Losses.device)
        for j in range(i, i+block_size):
            col_idx = j - i
            quant_int_w, quant_float_w = quantizer.quantize(W[:, j].unsqueeze(1))
            Q[:, j] = quant_float_w.squeeze(1)
            E[:, col_idx] = - (W[:, j] - Q[:, j]) / L[j, j]
            Losses[:, j] = E[:, col_idx] ** 2 / 2
            # (d_col, block_size - col_idx) = (d_col, 1) x（1, block_size - col_idx)
            block_delta = (E[:, col_idx]).unsqueeze(1) @ L[j, j:i+block_size].unsqueeze(0)
            W[:, j:i+block_size] += block_delta
        # (d_out, n - i - block_size) = (d_out, block_size) x (block_size, n - i - block_size)
        W[:, i+block_size:] += (E) @ L[i:i+block_size, i+block_size:]
    return Q, Losses, L.T


# ================= 使用 ====================
N, d_in, d_out = 1000, 32, 32
DEVICE = "cuda:0"
X = torch.randn(d_in, N).to(DEVICE)
W = torch.randn(d_out, d_in).to(DEVICE)
quantizer = Quantizer(4)
quantizer.get_quantizer_params(W)

Q_1, Losses_1 = gptq_1(X, W.clone(), N, d_in, d_out, quantizer)
print(W[:4, :4], Q_1[:4, :4], Losses_1.sum(), torch.sum((Q_1@X - W@X)**2 / N))
assert torch.allclose(Losses_1.sum(), torch.sum((Q_1@X - W@X)**2 / N))

Q_2, Losses_2, tracks_2 = gptq_2(X, W.clone(), N, d_in, d_out, quantizer)
print(W[:4, :4], Q_2[:4, :4], Losses_2.sum(), torch.sum((Q_2@X - W@X)**2 / N))
assert torch.allclose(Losses_2.sum(), torch.sum((Q_2@X - W@X)**2 / N))

Q_3, Losses_3, tracks_3 = gptq_3(X, W.clone(), N, d_in, d_out, quantizer)
print(W[:4, :4], Q_3[:4, :4], Losses_3.sum(), torch.sum((Q_3@X - W@X)**2 / N))
torch.allclose(Losses_3.sum(), torch.sum((Q_3@X - W@X)**2 / N))

# 三种算法的结果一致
assert torch.allclose(Losses_1, Losses_2)
assert torch.allclose(Losses_2, Losses_3, atol=1e-6)

# 验证 cholesky 分解矩阵 L 与迭代更新 H_inv 之间的联系
H = 2 / N * (X @ X.T)
H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
L = torch.linalg.cholesky(H_inv)
assert torch.allclose(tracks_2 @ tracks_2.T, H_inv)
assert torch.allclose(L @ L.T, H_inv)
assert torch.allclose(L, tracks_2)