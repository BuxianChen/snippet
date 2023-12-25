import torch
import torch.fx
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.rand(512, 32))
        self.linear = torch.nn.Linear(32, 64)

    def __getattr__(self, name):
        print(f"__getattr__({name})")
        return super().__getattr__(name)
    
    def forward(self, x, activate_fn):
        return self.linear(activate_fn(x).add(self.bias)[:3])

m = M()
print("Start Running")
m(torch.rand(512, 32), torch.nn.functional.relu)
print("End Running")

print("Start Tracing")
gm = symbolic_trace(m)
print("End Tracing")

g = gm.graph
print(g.python_code("self"))

g.print_tabular()