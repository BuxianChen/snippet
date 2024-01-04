# torch.quantizaion

`quant_methods_compare.py` 用于可视化 torch 原生支持的 3 种量化模式: dynamic quantization, static quantization, QAT. 脚本会展示一个简单的模型在量化过程中各个子模块的变化情况. 运行环境为 `torch==2.1.0`

```python
m = nn.Sequential(
    nn.Conv2d(2, 64, 8),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64, 64),
    nn.Linear(64, 10),
    nn.ReLU(),
    nn.LSTM(10, 10)
)
```

几种量化模式下子模块的变化情况如下

```
<<<dynamic quantization>>>
origin_float32_model              fused_float32_model                             dynamic_quantized_int8_model
--------------------------------  ----------------------------------------------  ----------------------------------------------------------------------
torch.nn.modules.conv.Conv2d      torch.nn.modules.conv.Conv2d                    torch.nn.modules.conv.Conv2d
torch.nn.modules.activation.ReLU  torch.nn.modules.activation.ReLU                torch.nn.modules.activation.ReLU
torch.nn.modules.flatten.Flatten  torch.nn.modules.flatten.Flatten                torch.nn.modules.flatten.Flatten
torch.nn.modules.linear.Linear    torch.nn.modules.linear.Linear                  torch.nn.modules.linear.Linear
torch.nn.modules.linear.Linear    torch.ao.nn.intrinsic.modules.fused.LinearReLU  torch.ao.nn.intrinsic.quantized.dynamic.modules.linear_relu.LinearReLU
torch.nn.modules.activation.ReLU  torch.nn.modules.linear.Identity                torch.nn.modules.linear.Identity
torch.nn.modules.rnn.LSTM         torch.nn.modules.rnn.LSTM                       torch.ao.nn.quantized.dynamic.modules.rnn.LSTM

<<<static quantization>>>
origin_float32_model              wrapped_float32_model                    fused_float32_model                             fused_float32_prepared_fp32_model               static_quantized_int8_model
--------------------------------  ---------------------------------------  ----------------------------------------------  ----------------------------------------------  --------------------------------------------------------------
<placeholder>                     torch.ao.quantization.stubs.QuantStub    torch.ao.quantization.stubs.QuantStub           torch.ao.quantization.stubs.QuantStub           torch.ao.nn.quantized.modules.Quantize
torch.nn.modules.conv.Conv2d      torch.nn.modules.conv.Conv2d             torch.nn.modules.conv.Conv2d                    torch.nn.modules.conv.Conv2d                    torch.ao.nn.quantized.modules.conv.Conv2d
torch.nn.modules.activation.ReLU  torch.nn.modules.activation.ReLU         torch.nn.modules.activation.ReLU                torch.nn.modules.activation.ReLU                torch.nn.modules.activation.ReLU
torch.nn.modules.flatten.Flatten  torch.nn.modules.flatten.Flatten         torch.nn.modules.flatten.Flatten                torch.nn.modules.flatten.Flatten                torch.nn.modules.flatten.Flatten
torch.nn.modules.linear.Linear    torch.nn.modules.linear.Linear           torch.nn.modules.linear.Linear                  torch.nn.modules.linear.Linear                  torch.ao.nn.quantized.modules.linear.Linear
torch.nn.modules.linear.Linear    torch.nn.modules.linear.Linear           torch.ao.nn.intrinsic.modules.fused.LinearReLU  torch.ao.nn.intrinsic.modules.fused.LinearReLU  torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU
torch.nn.modules.activation.ReLU  torch.nn.modules.activation.ReLU         torch.nn.modules.linear.Identity                torch.nn.modules.linear.Identity                torch.nn.modules.linear.Identity
torch.nn.modules.rnn.LSTM         torch.nn.modules.rnn.LSTM                torch.nn.modules.rnn.LSTM                       torch.ao.nn.quantizable.modules.rnn.LSTM        torch.ao.nn.quantized.modules.rnn.LSTM
<placeholder>                     torch.ao.quantization.stubs.DeQuantStub  torch.ao.quantization.stubs.DeQuantStub         torch.ao.quantization.stubs.DeQuantStub         torch.ao.nn.quantized.modules.DeQuantize

<<<qat static quantization>>>
origin_float32_model              wrapped_float32_model                    fused_float32_model                             fused_float32_prepared_fp32_model                         qat_static_quantized_int8_model
--------------------------------  ---------------------------------------  ----------------------------------------------  --------------------------------------------------------  --------------------------------------------------------------
<placeholder>                     torch.ao.quantization.stubs.QuantStub    torch.ao.quantization.stubs.QuantStub           torch.ao.quantization.stubs.QuantStub                     torch.ao.nn.quantized.modules.Quantize
torch.nn.modules.conv.Conv2d      torch.nn.modules.conv.Conv2d             torch.nn.modules.conv.Conv2d                    torch.ao.nn.qat.modules.conv.Conv2d                       torch.ao.nn.quantized.modules.conv.Conv2d
torch.nn.modules.activation.ReLU  torch.nn.modules.activation.ReLU         torch.nn.modules.activation.ReLU                torch.nn.modules.activation.ReLU                          torch.nn.modules.activation.ReLU
torch.nn.modules.flatten.Flatten  torch.nn.modules.flatten.Flatten         torch.nn.modules.flatten.Flatten                torch.nn.modules.flatten.Flatten                          torch.nn.modules.flatten.Flatten
torch.nn.modules.linear.Linear    torch.nn.modules.linear.Linear           torch.nn.modules.linear.Linear                  torch.ao.nn.qat.modules.linear.Linear                     torch.ao.nn.quantized.modules.linear.Linear
torch.nn.modules.linear.Linear    torch.nn.modules.linear.Linear           torch.ao.nn.intrinsic.modules.fused.LinearReLU  torch.ao.nn.intrinsic.qat.modules.linear_relu.LinearReLU  torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU
torch.nn.modules.activation.ReLU  torch.nn.modules.activation.ReLU         torch.nn.modules.linear.Identity                torch.nn.modules.linear.Identity                          torch.nn.modules.linear.Identity
torch.nn.modules.rnn.LSTM         torch.nn.modules.rnn.LSTM                torch.nn.modules.rnn.LSTM                       torch.ao.nn.quantizable.modules.rnn.LSTM                  torch.ao.nn.quantized.modules.rnn.LSTM
<placeholder>                     torch.ao.quantization.stubs.DeQuantStub  torch.ao.quantization.stubs.DeQuantStub         torch.ao.quantization.stubs.DeQuantStub                   torch.ao.nn.quantized.modules.DeQuantize
```