# from turtle import backward
import math

from numpy import isin

class Value:
    def __init__(self, data, _children=(), operation="", label=None):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self.ops = operation
        self._backward = lambda : None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        # Defining how gradient flows to the previous nodes
        def _backward():
            self.grad += 1 * out.grad # Chain rule
            other.grad += 1 * out.grad
        
        out._backward = _backward 
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")
        # Defining how gradient flows to the previous nodes
        def _backward():
            self.grad += other.data * out.grad # Chain rule 
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int,float)), "only support int/float powers"
        out = Value(self.data**other, (self, ), f"**{other}")
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad 
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)

    def tanh(self):
        x = self.data
        tanh_x = (math.exp(2*x) - 1 )/ (math.exp(2*x) + 1)
        out = Value(tanh_x, (self, ), "tanh")
        def _backward():
            ### 1 - tanh(x)**2 is the Differential of tanh(x) w.r.t x
            self.grad += (1 - tanh_x**2) * out.grad # Chain rule
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,) , "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward()
        return out

    def backward(self):
        visited = set()
        topo_sorted_nodes = []
        def build_topo(x):
            if x not in visited:
                visited.add(x)
                for child in x._prev:
                    build_topo(child)
                topo_sorted_nodes.append(x)
        build_topo(self)

        self.grad = 1

        for node in reversed(topo_sorted_nodes):
            node._backward()
        

            