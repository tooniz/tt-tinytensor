import torch
import torch.nn.functional as functional
from 
tt_path_flag = False

def matmul(lin, rin, tt_runtime = None):
    # assume minimum dimension fits into both cores and max amount of dram queues
    # for now...
    min_dim = min(lin.shape[-2], min(lin.shape[-1], rin.shape[-1]))
    row_fold = int(lin.shape[-2] / min_dim)
    id_fold = int(lin.shape[-1] / min_dim)
    col_fold = int(rin.shape[-1] / min_dim)
    lin_folded, rin_folded = fold_for_matmul(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
    if(tt_runtime == None):
        out = torch.matmul(lin_folded, rin_folded)
        out = sum(out,-3)
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)
    golden = torch.matmul(lin,rin)
    assert torch.allclose(out, golden, atol=0.005, rtol=0.005)
    return out

def sum(tensor, dim, tt_runtime=None):
    if(tt_runtime == None):
        out = torch.sum(tensor, dim)
    else:
        # swap sum dimension and columns
        tens = tensor.swapaxes(-1,dim)
        ones = tt_tensor()
        tens = matmul(tens,ones)
        pass
    return out

def add(linput, rinput, tt_runtime: None):
    out = linput + rinput
    return out

def linear(acts, weights, bias, tt_runtime = None):
    out = matmul(acts,weights, tt_runtime)
    out = add(out, bias, tt_runtime)
    return out

def softmax(input, dim, output=None):
    out = functional.softmax(input, dim)
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


######
# Helper functions

def make_expand_mask(shape_list, dim, factor):
    expand_dim = shape_list[dim]
    out_list = shape_list
    for i in range(len(shape_list)):
        out_list[i] = -1
    out_list[dim] = expand_dim * factor
    return out_list

def fold_for_matmul(linput, rinput, rowfactor, colfactor, idfactor):
    # check basic assumptions
    assert linput.shape[-1] == rinput.shape[-2]
    assert linput.shape[-1] % idfactor == 0
    assert linput.shape[-2] % rowfactor == 0
    assert rinput.shape[-1] % colfactor == 0
    # expand to equal dimensions before folding
    if(sum(list(linput.shape)) > sum(list(rinput.shape))):
        rrshp = rinput.broadcast_to(linput.shape)
    elif(sum(list(rinput.shape)) > sum(list(linput.shape))):
        lrshp = linput.broadcast_to(linput.shape)
    # figure out reshaped dims
    out_cols = int(rinput.shape[-1] / colfactor)
    out_rows = int(linput.shape[-2] / rowfactor)
    out_id = int(linput.shape[-1] / idfactor)
    #reshape
    l_shape = list(linput.shape)
    r_shape = list(rinput.shape)
    # pop off the row and column dimensions
    l_shape.pop()
    l_shape.pop()
    r_shape.pop()
    r_shape.pop()
    # add the reshaped bottom dimensions
    l_shape.extend([rowfactor,out_rows,idfactor,out_id])
    r_shape.extend([idfactor,out_id,colfactor,out_cols])
    # reshape
    lrshp = linput.reshape(l_shape)
    rrshp = rinput.reshape(r_shape)
    # swap axes to make r,c at bottom
    lrshp = lrshp.swapaxes(-2,-3)
    rrshp = rrshp.swapaxes(-2,-3)
    rrshp = rrshp.swapaxes(-3,-4)
    # add extra dims for broadcasting to each others shape
    lrshp = lrshp.unsqueeze(-4)
    rrshp = rrshp.unsqueeze(-5)
    l_expand = make_expand_mask(list(lrshp.shape),-4,rrshp.shape[-4])
    r_expand = make_expand_mask(list(rrshp.shape),-5,lrshp.shape[-5])
    lrshp = lrshp.expand(l_expand)
    rrshp = rrshp.expand(r_expand)
    # broadcast to each others shapes
    l_shape = lrshp.shape
    r_shape = rrshp.shape
    lrshp = lrshp.broadcast_to(r_shape)
    rrshp = rrshp.broadcast_to(l_shape)
    return lrshp, rrshp

def unfold_output(input, rowfactor, colfactor):
    shape = list(input.shape)
    c = shape[-1]
    r = shape[-2]
    for _ in range(4):
        shape.pop()
    mc = c * colfactor
    mr = r * rowfactor
    shape.append(mr)
    shape.append(mc)
    return input.swapaxes(-2,-3).reshape(shape)



