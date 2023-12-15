import torch
from dataclasses import dataclass
from typing import Tuple, Dict, Union, Any
from types import FunctionType
from functools import wraps
import inspect

@dataclass
class Node:
    opcode: str
    target: Union[FunctionType, str]
    name: str
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    # meta: Optional[str]  # TODO: 便于实现 Graph.python_code 用, 暂时只写在注释里, 后续看怎么弄

    # 用于 Graph.python_code
    def __repr__(self):
        return self.name


class Proxy:
    node: Node
    tracer: "Tracer"

    def __init__(self, node, tracer):
        self.node = node
        self.tracer = tracer

    def __torch_function__(self, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        if torch.overrides.is_tensor_method_or_property(orig_method):  # torch.Tensor.add 进入此分支
            return self.tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)  # , meta="torch.Tensor"
        else:  # torch.add 进入此分支
            return self.tracer.create_proxy('call_function', orig_method, args, kwargs)
    
    def __getattr__(self, k):
        return Attribute(self, k)

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", '__call__', (self,) + args, kwargs)  # , meta="self"

    def __add__(self, other):
        return self.tracer.create_proxy("call_function", "operator.add", (self, other), {})

    def __radd__(self, other):
        return self.tracer.create_proxy("call_function", "operator.add", (other, self), {})

# TODO: 需要理解这个继承
class Attribute(Proxy):
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node = None
    
    @property
    def node(self):
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)  # , meta="self"


class Graph:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes else []
    
    def insert(self, n: Node):
        self.nodes.append(n)

    @staticmethod
    def _format_args(args, kwargs):
        args_s = ', '.join(repr(a) for a in args)
        kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
        if args_s and kwargs_s:
            return f'{args_s}, {kwargs_s}'
        return args_s or kwargs_s
    
    def python_code(self):
        codes = []
        input_args = []
        for node in self.nodes:
            if node.opcode == "placeholder":
                codes.append(f"{node.name} = {node.target}")
                input_args.append(node.target)
            elif node.opcode == "call_method":
                # TODO:  xxx.{node.target}({self._format_args(args[1:], kwargs)})
                # xxx 怎么确定?
                codes.append(f"++call_method++, {str(node)}")
            elif node.opcode == "call_function":
                if isinstance(node.target, str) and node.target.startswith("operator"):
                    # TODO: 可能还是得像原始实现那样用字典在外部动态添加Proxy的魔术方法
                    codes.append("{} = {} + {}".format(node.name, *(repr(arg) for arg in node.args)))
                else:
                    # TODO: getattr 与 torch.add 这类
                    codes.append(f"++call_function++, {str(node)}")
            elif node.opcode == "get_attr":
                codes.append(f"{node.name} = self.{node.target}")
            elif node.opcode == "call_module":
                # TODO: 小BUG, 基本上不会出现: 可能是直接 self(...)
                codes.append(f"{node.name} = self.{node.target}({Graph._format_args(node.args, node.kwargs)})")
            elif node.opcode == "output":
                # TODO: 这里是否有多个返回值的情况?
                codes.append(f"return {repr(node.args[0])}")
        header = ", ".join(input_args)
        header = "def forward(self, {header})\n"
        code_str = header + "\n".join(["    "+code for code in codes])
        return code_str


class Tracer:
    def __init__(self):
        self._node_idx = 0

    def create_arg(self, a):
        if isinstance(a, (tuple, list)):
            return type(a)(self.create_arg(x) for x in a)
        if isinstance(a, dict):
            return {k: self.create_arg(v) for k, v in a.items()}
        # 以下情形均为基本类型
        if isinstance(a, Proxy):
            return a.node
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        if isinstance(a, torch.Tensor):
            for n, p in self.root.named_buffers():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            # 为简单起见: 假设 self.root 中没有不在 _buffers 中的 tensor 属性
        return a
    
    def create_node(self, opcode, target, args, kwargs):  # , meta=None
        n = Node(opcode=opcode, target=target, args=args, kwargs=kwargs, name=f"node_{self._node_idx}")  # , meta=meta
        self._node_idx += 1
        self.graph.insert(n)
        return n
    
    def create_proxy(self, opcode, target, args, kwargs):  # , meta=None
        args_: Tuple[Union[Node, Any]] = self.create_arg(args)
        kwargs_: Dict[str, Union[Node, Any]] = self.create_arg(kwargs)
        return Proxy(self.create_node(opcode, target, args_, kwargs_), self)  # , meta

    def trace(self, m: torch.nn.Module):
        self.graph = Graph()
        self.root = m
        fn = self.root.forward  # 为简单起见, 假设 forward 函数不包含可变参数

        sig = inspect.signature(fn)
        co = fn.__code__
        total_argcount = co.co_argcount + co.co_kwonlyargcount
        names = co.co_varnames

        args = []
        for i in range(1, total_argcount):
            args.append(self.create_proxy("placeholder", names[i], (), {}))

        
        orig_module_call = torch.nn.Module.__call__
        orig_module_getattr = torch.nn.Module.__getattr__
        
        @wraps(orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            for n, m in self.root.named_modules():
                if mod is m:
                    path = n
                    break
            else:
                raise NameError("submodule not found")
            is_leaf = mod.__module__.startswith("torch.nn") and not isinstance(mod, torch.nn.Sequential)

            if is_leaf:
                return self.create_proxy("call_module", path, args, kwargs)
            else:
                return orig_module_call(mod, *args, **kwargs)

        @wraps(orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = orig_module_getattr(mod, attr)
            if isinstance(attr_val, torch.nn.Parameter):
                for n, p in self.root.named_parameters():
                    if attr_val is p:
                        return self.create_proxy('get_attr', n, (), {})
                raise NameError('parameter is not a member of this module')
            if isinstance(attr_val, torch.Tensor):
                for n, p in self.root.named_buffers():
                    if attr_val is p:
                        return self.create_proxy('get_attr', n, (), {})
                raise NameError('buffer tensor is not a member of this module')
            return attr_val


        setattr(torch.nn.Module, "__call__", module_call_wrapper)
        setattr(torch.nn.Module, "__getattr__", module_getattr_wrapper)

        self.create_node("output", "output", (self.create_arg(fn(*args)),), {})

        setattr(torch.nn.Module, "__call__", orig_module_call)
        setattr(torch.nn.Module, "__getattr__", orig_module_getattr)

        return self.graph


if __name__ == "__main__":
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(torch.sum(
                self.linear(x + self.linear.weight).relu(), dim=-1), 3)

    m = MyModule()
    tracer = Tracer()
    g = tracer.trace(m)
    for node in g.nodes:
        print(node)
        print()
    print(g.python_code())
