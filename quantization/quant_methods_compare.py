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

# 获取 layer 的完整名称
def get_full_name(layer):
    class_name = type(layer).__name__
    module_name = type(layer).__module__
    full_name = module_name + '.' + class_name
    return full_name

# 判断是否有 hook
def has_hook(layer):
    from torch.ao.quantization.quantize import _observer_forward_hook
    from torch.ao.quantization.observer import ObserverBase
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    from torch.ao.quantization.stubs import QuantStub, DeQuantStub

    """
        static quantization:
            在 prepare 之后, 对于一个layer,
            如果 layer 需要被量化或是 QuantStub 层, 那么它包含以下属性
                activation_post_process: ObserverBase, 它被注册为 layer 的 forward_hook, 用于观测 layer 层输出的取值范围
            如果 layer 是 DeQuantStub, 那么它什么都不包含
            如果 layer 无需量化, 则全流程保持为普通的 float 形式的 nn.Module, 且不会追加任何 hook 或子模块
        qat:
            在 prepare_qat 之后, 对于一个layer,
            如果 layer 需要被量化, 那么它包含如下属性
                weight_fake_quant: FakeQuantizeBase,   # 在 layer 的 forward 函数中被调用, 用于对权重的量化与反量化
                weight_fake_quant.activation_post_process: ObserverBase,  # 在 layer.weight_fake_quant 的 forward 函数中被调用, 用于观测权重的取值范围
                activation_post_process: FakeQuantizeBase,  # 被注册为 layer 的 forward_hook, 用于对 layer 层输出进行量化与反量化
                activation_post_process.activation_post_process: ObserverBase  # 在 layer.activation_post_process 的 forward 函数中被调用, 用于观测 layer 层输出的取值范围
            如果 layer 是 QuantStub, 那么它只包含 activation_post_process 和 activation_post_process.activation_post_process
            如果 layer 是 DeQuantStub, 那么它什么都不包含
            如果 layer 无需量化, 则全流程保持为普通的 float 形式的 nn.Module, 且不会追加任何 hook 或子模块
    """

    hook_names = [
        "_backward_pre_hooks", "_backward_hooks",
        "_forward_hooks", "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs",
    ]

    hook_dict = {hook_name: getattr(layer, hook_name) for hook_name in hook_names if getattr(layer, hook_name)}
    if hook_dict:
        # 确保只有一个 _forward_hook (注意这种hook是在forward之后被调用的) 且 hook 为 _observer_forward_hook 且相应的 layer 包含 activation_post_process
        assert (
            len(hook_dict) == 1 and "_forward_hooks" in hook_dict
            and len(hook_dict["_forward_hooks"]) == 1
            and list(hook_dict["_forward_hooks"].values())[0] is _observer_forward_hook
            and hasattr(layer, "activation_post_process")
        )

        # static quantization: activation_post_process 是 ObserverBase
        if isinstance(layer.activation_post_process, ObserverBase):
            return "static"
        # qat: activation_post_process 是 FakeQuantizeBase, 并且它的 forward 方法包含了对 observer 的调用, 而 activation_post_process 本身被注册为了 forward_hook
        elif isinstance(layer.activation_post_process, FakeQuantizeBase):
            assert (
                hasattr(layer.activation_post_process, "activation_post_process")
                and not has_hook(layer.activation_post_process)
            )
            if isinstance(layer, QuantStub):
                assert not hasattr(layer, "weight_fake_quant")
            else:
                assert hasattr(layer, "weight_fake_quant") and not has_hook(layer.weight_fake_quant)
            return "qat"
        else:
            raise ValueError("other case")
    else:
        return ""

# 如果有 hook, 则在 layer 的 classname 前面加上 *
def get_display_name(layer):
    full_name = get_full_name(layer)
    flag_mapping = {
        "qat": "+",
        "static": "*"
    }
    case = has_hook(layer)
    if case:
        full_name = flag_mapping[case]+full_name
    return full_name

# 展平带有 QuantStub 和 DeQuantStub 被 QuantWrapper 包裹的 module
def flatten_submodule(m):
    return [m.quant] + list(m.module) + [m.dequant]


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
        list(map(get_display_name, item))
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

    column_names = ["origin_float32_model", "wrapped_float32_model", "fused_float32_model", "fused_float32_prepared_fp32_model", "static_quantized_int8_model"]
    
    module_names = [
        list(map(get_display_name, item))
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

    column_names = ["origin_float32_model", "wrapped_float32_model", "fused_float32_model", "fused_float32_prepared_fp32_model", "qat_static_quantized_int8_model"]
    
    module_names = [
        list(map(get_display_name, item))
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