import torch
import torch.nn.functional as functional

tt_path_flag = False
def linear(acts, weights, bias, output=None):
    out = functional.linear(acts,weights,bias)
    if(output != None):
        output.copy_(out)
    return out

def softmax(input, dim, output=None):
    out = functional.softmax(input, dim)
    if(output != None):
        output.copy_(out)
    return out

def add(linput, rinput, output=None):
    out = linput + rinput
    if(output != None):
        output.copy_(out)
    return out

def mul(linput, rinput, output=None):
    out = linput * rinput
    if(output != None):
        output.copy_(out)
    return out

def gelu(input, output=None):
    out = functional.gelu(input)
    if(output != None):
        output.copy_(out)
    return out

def relu(input, output=None):
    out = functional.relu(input)
    if(output != None):
        output.copy_(out)
    return out

def sqrt(input, output=None):
    out = torch.sqrt(input, dim)
    if(output != None):
        output.copy_(out)
    return out

def reciprocal(input, output=None):
    out = 1/input
    if(output != None):
        output.copy_(out)
    return out

def reduce_sum(input, dim, output=None):
    out = torch.sum(input, dim)
    if(output != None):
        output.copy_(out)
    return out

def reduce_max(input, dim, output=None):
    out = torch.max(input, dim)
    if(output != None):
        output.copy_(out)
    return out

######
# Shape manipulation
def view_local(input, view_spec):
    if(not tt_path_flag):
        return input.view(view_spec)

def permute_local(input, permute_spec):
    if(not tt_path_flag):
        return input.permute(permute_spec)
