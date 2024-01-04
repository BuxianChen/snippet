import torch
from torch import nn
from tabulate import tabulate


def get_example_model():
    m = nn.Sequential(
        nn.Conv2d(2, 64, 8),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64, 64),
        nn.Linear(64, 10),
        nn.ReLU(),
        nn.LSTM(10, 10)
    )
    return m

def get_full_name(layer):
    class_name = type(layer).__name__
    module_name = type(layer).__module__
    full_name = module_name + '.' + class_name
    return full_name


def dynamic_quantization_layer_change(origin_float32_model):
    import torch.ao.nn.intrinsic as nni
    origin_float32_model.eval()
    # 需要手动指定哪几层可以 fuse, 这也是 fx 模式的优势
    fused_float32_model = torch.ao.quantization.fuse_modules(origin_float32_model, [["4", "5"]], inplace=False)
    # 可以手动或者用默认的 qconfig_spec
    dynamic_quantized_int8_model = torch.ao.quantization.quantize_dynamic(
        model=fused_float32_model, qconfig_spec={nn.LSTM, nni.LinearReLU}, dtype=torch.qint8, inplace=False
    )

    column_names = ["origin_float32_model", "fused_float32_model", "dynamic_quantized_int8_model"]
    module_names = [
        list(map(get_full_name, item))
        for item in zip(origin_float32_model, fused_float32_model, dynamic_quantized_int8_model)
    ]
    modules = [
        origin_float32_model,
        fused_float32_model,
        dynamic_quantized_int8_model
    ]
    return module_names, column_names, modules

def static_quantization_layer_change(origin_float32_model):
    from torch.ao.quantization import QuantWrapper
    
    # 1. 此处需要将 QuantStub 与 DeQuantStub 追加到原始模型上, 并在 forward 前后追加这两个层的逻辑
    # 2. 并且有可能需要用 torch.ao.nn.quantized.FloatFunctional 代替 forward 中的加法运算符等操作
    wrapped_float32_model = QuantWrapper(origin_float32_model)  # children: "quant", "dequant", "module"

    wrapped_float32_model.eval()
    wrapped_float32_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # fuse model
    fused_float32_model = torch.ao.quantization.fuse_modules(wrapped_float32_model, [['module.4', 'module.5']])

    # prepare model: 添加 observer
    fused_float32_prepared_fp32_model = torch.ao.quantization.prepare(fused_float32_model)

    # 送入校准数据使得 observer 观测到每一层激活值的分布
    input_fp32 = torch.randn(4, 2, 8, 8)
    fused_float32_prepared_fp32_model(input_fp32)

    # 根据 observer 确定激活值的量化参数, 并量化模型本身的权重
    static_quantized_int8_model = torch.ao.quantization.convert(fused_float32_prepared_fp32_model)

    def flatten_submodule(m):
        return [m.quant] + list(m.module) + [m.dequant]

    column_names = ["origin_float32_model", "wrapped_float32_model", "fused_float32_model", "fused_float32_prepared_fp32_model", "static_quantized_int8_model"]
    
    module_names = [
        list(map(get_full_name, item))
        for item in zip(
            [torch.nn.Identity()] + list(origin_float32_model) + [torch.nn.Identity()],
            flatten_submodule(wrapped_float32_model),
            flatten_submodule(fused_float32_model),
            flatten_submodule(fused_float32_prepared_fp32_model),
            flatten_submodule(static_quantized_int8_model)
        )
    ]
    module_names[0][0] = "<placeholder>"
    module_names[-1][0] = "<placeholder>"
    modules = [
        origin_float32_model,
        wrapped_float32_model,
        fused_float32_model,
        fused_float32_prepared_fp32_model,
        static_quantized_int8_model
    ]
    return module_names, column_names, modules

def qat_layer_change(origin_float32_model):
    from torch.ao.quantization import QuantWrapper
    
    # 1. 此处需要将 QuantStub 与 DeQuantStub 追加到原始模型上, 并在 forward 前后追加这两个层的逻辑
    # 2. 并且有可能需要用 torch.ao.nn.quantized.FloatFunctional 代替 forward 中的加法运算符等操作
    wrapped_float32_model = QuantWrapper(origin_float32_model)  # children: "quant", "dequant", "module"

    wrapped_float32_model.eval()
    # !! 注意这里与 static quantization 的不同
    wrapped_float32_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    # fuse model
    # !! 注意这里与 static quantization 的不同
    fused_float32_model = torch.ao.quantization.fuse_modules_qat(wrapped_float32_model, [['module.4', 'module.5']])

    # prepare model: 添加 observer
    fused_float32_model.train()  # 必须先设置为 train 模式再使用 prepare_qat
    # !! 注意这里与 static quantization 的不同
    fused_float32_prepared_fp32_model = torch.ao.quantization.prepare_qat(fused_float32_model)

    # 送入校准数据使得 observer 观测到每一层激活值的分布
    input_fp32 = torch.randn(4, 2, 8, 8)
    fused_float32_prepared_fp32_model(input_fp32)

    # 根据 observer 确定激活值的量化参数, 并量化模型本身的权重
    qat_static_quantized_int8_model = torch.ao.quantization.convert(fused_float32_prepared_fp32_model)

    def flatten_submodule(m):
        return [m.quant] + list(m.module) + [m.dequant]

    column_names = ["origin_float32_model", "wrapped_float32_model", "fused_float32_model", "fused_float32_prepared_fp32_model", "qat_static_quantized_int8_model"]
    
    module_names = [
        list(map(get_full_name, item))
        for item in zip(
            [torch.nn.Identity()] + list(origin_float32_model) + [torch.nn.Identity()],
            flatten_submodule(wrapped_float32_model),
            flatten_submodule(fused_float32_model),
            flatten_submodule(fused_float32_prepared_fp32_model),
            flatten_submodule(qat_static_quantized_int8_model)
        )
    ]
    module_names[0][0] = "<placeholder>"
    module_names[-1][0] = "<placeholder>"
    modules = [
        origin_float32_model,
        wrapped_float32_model,
        fused_float32_model,
        fused_float32_prepared_fp32_model,
        qat_static_quantized_int8_model
    ]
    return module_names, column_names, modules


if __name__ == "__main__":
    m = get_example_model()
    module_names, column_names, modules = dynamic_quantization_layer_change(m)
    print("<<<dynamic quantization>>>")
    print(tabulate(module_names, headers=column_names))
    print()

    module_names, column_names, modules = static_quantization_layer_change(m)
    print("<<<static quantization>>>")
    print(tabulate(module_names, headers=column_names))
    print()

    module_names, column_names, modules = qat_layer_change(m)
    print("<<<qat static quantization>>>")
    print(tabulate(module_names, headers=column_names))
    print()