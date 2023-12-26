**示例 1: 正确理解 `Tracer.trace` 的执行逻辑**

```python
import torch
import torch.fx
from torch.fx import symbolic_trace, Tracer
import dis

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.rand(512, 32))
        self.linear = torch.nn.Linear(32, 64)

    def __getattr__(self, name):
        print(f"__getattr__({name})")
        return super().__getattr__(name)
    
    def forward(self, x, activate_fn):
        return self.linear(activate_fn(x).add(self.bias))

m = M()
tracer = Tracer()
print("Start Running")
m(torch.rand(512, 32), torch.nn.functional.relu)
print("End Running")

print("Start Tracing")
g = tracer.trace(m)
print("End Tracing")

print(g.python_code("self"))

g.print_tabular()

dis.dis(m.forward)
```

运行环境为 `Python 3.8, torch==1.8.0`, 输出结果为

```
__getattr__(bias)
Start Running
__getattr__(linear)
__getattr__(bias)
End Running
Start Tracing
__getattr__(linear)
__getattr__(bias)
End Tracing

def forward(self, x, activate_fn):
    call = activate_fn.__call__(x);  activate_fn = x = None
    bias = self.bias
    add_1 = call.add(bias);  call = bias = None
    linear_1 = self.linear(add_1);  add_1 = None
    return linear_1
    
opcode       name         target       args              kwargs
-----------  -----------  -----------  ----------------  --------
placeholder  x            x            ()                {}
placeholder  activate_fn  activate_fn  ()                {}
call_method  call         __call__     (activate_fn, x)  {}
get_attr     bias         bias         ()                {}
call_method  add_1        add          (call, bias)      {}
call_module  linear_1     linear       (add_1,)          {}
output       output       output       (linear_1,)       {}

 17           0 LOAD_FAST                0 (self)
              2 LOAD_METHOD              0 (linear)
              4 LOAD_FAST                2 (activate_fn)
              6 LOAD_FAST                1 (x)
              8 CALL_FUNCTION            1
             10 LOAD_METHOD              1 (add)
             12 LOAD_FAST                0 (self)
             14 LOAD_ATTR                2 (bias)
             16 CALL_METHOD              1
             18 CALL_METHOD              1
             20 RETURN_VALUE
```

注意点 1:

Proxy 的 `__getattr__` 方法返回 `Attribute` 对象, 并且暂时不触发 `create_node`, 否则的话可能会多创造无用节点. 这一点可以通过在 `Attribute` 的 `__init__` 方法的结尾加上一行 `self.node` 来进行调试, 观察 `g.python_code("self")` 输出结果的变化

注意点 2:

依据 `dis.dis` 的输出, 准确理解其执行逻辑如下:

- 0, 2: 首先 `self.linear` 触发 `nn.Module.__getattr__` 方法(因为 `linear` 存储在 `self._modules` 中, 而 `nn.Module` 没有重写 `__getattribute__` 方法, 所以会落到对 `__getattr__` 的调用), 注意此方法在 trace 之前被替换掉了, 但这里因为 `self.linear` 不是 `nn.Parameter` 类型, 所以跟被替换前的调用没有区别, 因此这过程也不会触发 `create_node` 或 `create_proxy`
- 4, 6, 8: 然后直接执行 `activate_fn(x)`, 这里因为 `activate_fn` 被包装成了 `Proxy`, 因此直接会触发 `Proxy.__call__` 的调用, 最终返回一个 `Proxy` 对象, 此过程中触发 `create_node("call_method", "__call__", ...)`
- 10: 上一步返回的 `Proxy` 获取 `add` 方法, 此时也会触发 `Proxy.__getattr__` 方法的调用, 得到一个 `Attribute` 对象
- 12, 14: 正常执行 `nn.Module.__getattr__` 方法返回 `self.bias`, 因为 `self.bias` 是 `nn.Tensor` 类型而不是 `nn.Parameter` 类型, 因此与第 `0, 2` 行相同, 与原本被替换前的 `__getattr__` 方法没有区别
- 16: 此时触发第 `10` 行返回的 `Attribute` 对象的 `__call__` 方法, 而这个方法会先触发对第 `12, 14` 行返回结果关于 `Tracer.create_arg` 的调用, 进入分支 `create_node("get_attr", ...)`, 随后继续执行 `Attribute.__call__` 即 `create_node("call_method", ...)`
- 18: 第 `0, 2` 行的返回结果对 第 `16` 行返回结果的调用, 这时因为 `self.linear` 是 `nn.Module` 类型, 而 `nn.Module` 类型在 `trace` 之前被替换掉了, 因此执行被替换掉的方法, 即回落到 `Tracer.call_module` 的调用, 因为 `self.linear` 是 `nn.Linear` 类型, 所以是叶子节点 (`Tracer.is_leaf_module(self.linear)` 为 `True`), 因此触发 `create_proxy("call_module", ...)`, 因为此时参数是第 `16` 行返回的 `Proxy` 对象, 所以先触发的 `create_arg` 直接返回 `Proxy.node`, 然后直接触发 `create_node("call_module", ...)`
