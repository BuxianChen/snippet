import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic


# =================== linear weight 量化 =========================
# weight: dtype=torch.qint8, symmetric=True, reduce_range=False
# input: 约为: dtype=torch.quint8, symmetric=False, reduce_range=True
def calculate_qparams_linear_weight(weight, dtype=torch.qint8, symmetric=True, reduce_range=False):
    # dtype: qint8/quint8, symmetric: False/True
    # only support per tensor
    min_val, max_val = weight.min(), weight.max()
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))  # 用了 0.0 截断
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))  # 用了 0.0 截断

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    if reduce_range:
        if dtype == torch.qint8:
            quant_min, quant_max = -64, 63
        elif dtype == torch.quint8:
            quant_min, quant_max = 0, 127
        # else: raise ValueError(f"{dtype} not support")
    else:
        if dtype == torch.qint8:
            quant_min, quant_max = -128, 127
        elif dtype == torch.quint8:
            quant_min, quant_max = 0, 255
        # else: raise ValueError(f"{dtype} not support")

    if symmetric:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        # symmetric 情况下, 0.0 的量化表示是 0/128
        if dtype == torch.quint8:
            zero_point = zero_point.new_full(zero_point.size(), 128)
        # elif dtype == torch.qint8:
        #     zero_point = zero_point.new_full(zero_point.size(), 0)
        # else: raise ValueError(f"{dtype} not support")
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # 注意到这可能会使得 min_val_neg 或 max_val_neg 对应的量化值可能不是 quant_min 或 quant_max
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int64)  # 必须用 round 取整, 如果直接用 to(torch.int) 会是向 0 取整
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point, quant_min, quant_max

def quantize_linear_weight(weight, dtype=torch.qint8, symmetric=True, reduce_range=False):
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(
        weight, dtype=dtype, symmetric=symmetric, reduce_range=reduce_range)
    qweight = torch.round(weight / scale).to(torch.int64) + zero_point
    qweight = torch.clamp(qweight, quant_min, quant_max)
    return qweight

def dequantize_linear_weight(qweight, scale, zero_point, quant_min, quant_max):
    deqweight = (qweight - zero_point).float() * scale
    return deqweight

# torch 默认用 symmetric 的方式量化权重 
def torch_dynamic_quantize_linear_weight(layer: torch.nn.Linear):
    qmodel = quantize_dynamic(model=torch.nn.Sequential(layer), qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)
    weight = qmodel[0].weight()
    
    zero_point = weight.q_zero_point()  # int scaler
    scale = weight.q_scale()            # float scalar
    qweight = weight.int_repr().long()  # int64 tensor
    return qweight, scale, zero_point

# 单元测试
def test_linear_weight_quantize(layer: torch.nn.Linear):
    dtype = torch.qint8
    symmetric = True
    weight = layer.weight
    scale_m, zero_point_m, _, _ = calculate_qparams_linear_weight(weight, dtype=dtype, symmetric=symmetric)
    qweight_m = quantize_linear_weight(weight, dtype=dtype, symmetric=symmetric)

    qweight_t, scale_t, zero_point_t = torch_dynamic_quantize_linear_weight(layer)

    assert int(zero_point_m) == int(zero_point_t)
    assert float(scale_m) - float(scale_t) < 1e-7
    assert (qweight_m - qweight_t).abs().sum().item() == 0

# ========================= linear input 量化 ============================
# https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31
# 基本上类似于 quantize_linear_weight: symmetric = False, dtype=quint8, reduce_range=True
# 但 zero_point 和 scale 的计算与weight有所区别 (主要是zero_point会在min_val或max_val里选择进行计算)
# TODO: 为简单起见, 这里忽略 zero_point 的差异, 精确实现要参照上面链接里的源码: auto q_params = quant_utils::ChooseQuantizationParams(...)
def quantize_linear_input(x, dtype=torch.quint8, symmetric=False, reduce_range=True):
    return quantize_linear_weight(x, dtype=dtype, symmetric=symmetric, reduce_range=reduce_range)

# ======================== 量化数据的矩阵乘法实现(仅示意) ===========================
def gemm(qweight, qinput, bias, weight_qparams, input_qparams):
    # input @ weight.T + bias, qweight(n*k), qinput(m*k), bias(n)
    # 计算公式如下:
    # deq_x = (q_x - z_x) * s_x, deq_w.T = (q_w.T - z_w.T) * s_w (note: z_x, z_w is matrix)
    # output = deq_x @ deq_w.T + bias = s_w * s_x * (q_x @ q_w.T - q_x @ z_w .T- z_x @ q_w.T + z_x@z_w.T) + bias
    # 注意在真实的实现里: s_w * s_x, z_x @ q_w.T, z_x@z_w 其实可以提前算好, 只有 q_x @ q_w.T 和 q_x @ z_w 需要计算
    m, k = qinput.shape
    n = qweight.shape[0]

    s_w, z_w = weight_qparams
    s_x, z_x = input_qparams

    # 这里比较低效, 直接创建了常量张量
    z_w = torch.ones_like(qweight) * z_w
    z_x = torch.ones_like(qinput) * z_x

    # 真正的实现是用 torch.int32 进行累加, 此处为了简单 qweight, qinput, buffer 其实都用了 torch.long 类型
    buffer = torch.zeros((m, n), dtype=torch.int64)
    output = torch.zeros((m, n), dtype=torch.float64)  # 真实实现(fbgemm)用double类型

    buffer += qinput @ qweight.T
    buffer -= qinput @ z_w.T
    buffer -= z_x @ qweight.T  # 可提前算好
    buffer += z_x @ z_w.T      # 可提前算好

    output = buffer.double() * s_w * s_x + bias
    output = output.to(torch.float32)  # 真实实现里, 这里是回到 input.dtype
    return output


def torch_dynamic_quantization_forward(layer, inp):
    # fbgemm: per tensor quantize weight (qint8) and input (quint8)
    qmodel = quantize_dynamic(model=torch.nn.Sequential(layer), qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)
    output = qmodel(inp)
    return output


# 不使用任何 pytorch quantization API 的实现
def manual_dynamic_quantization_forward(layer, inp):
    weight = layer.weight
    bias = layer.bias
    # step 1 权重量化, 参考: torch.ao.quantization.observer.MinMaxObserver, 注意取整的方式(四舍五入 vs 向零取整)
    qweight = quantize_linear_weight(weight, symmetric=True)
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(weight, symmetric=True)
    weight_qparams = (scale, zero_point)
    # deqweight = dequantize_linear_weight(qweight, scale, zero_point, quant_min, quant_max)
    
    # step 2 输入量化, 参考: C++ 源码: https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L31
    qinput = quantize_linear_weight(inp, dtype=torch.quint8, symmetric=False, reduce_range=True)
    scale, zero_point, quant_min, quant_max = calculate_qparams_linear_weight(inp, dtype=torch.quint8, symmetric=False, reduce_range=True)
    input_qparams = (scale, zero_point)
    # deqinput = dequantize_linear_weight(qinput, scale, zero_point, quant_min, quant_max)

    # step 3 量化后的输入与量化后权重进行计算并还原为 float32, 并加上偏置项
    output = gemm(qweight, qinput, bias, weight_qparams, input_qparams)
    return output

if __name__ == "__main__":
    batch_size, in_features, out_features = 20, 30, 40
    layer = torch.nn.Linear(in_features, out_features)
    inp = torch.distributions.normal.Normal(0, 1).sample((batch_size, in_features))
    # 测试权重量化的正确性
    test_linear_weight_quantize(layer)

    # 测试手工实现与pytorch的差异
    torch_output = torch_dynamic_quantization_forward(layer, inp)
    manual_output = manual_dynamic_quantization_forward(layer, inp)
    print("测试手工实现与pytorch的差异:", (torch_output - manual_output).abs().max().item())