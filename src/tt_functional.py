import torch
import torch.nn.functional as functional
from tt_tensor import tt_tensor
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
from tt_netlist import tt_netlist
from tt_netlist import tt_net_op_types

# def matmul(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None):
#     if(runtime == None):
#         out = torch.matmul(lin,rin)
#     else:
#         # if all dimensions fit onto chip, and don't violate max dram queues - just run
#         # without folding
#         if(lin.shape[-2] < runtime.simd_cluster.r_cores and rin.shape[-1] < runtime.simd_cluster.c_cores and rin.shape[-2] < runtime.simd_cluster.queue_lim):
#             row_fold = 1
#             col_fold = 1
#             id_fold = 1
#             out = runtime.netlist.binary_tensor_op(tt_net_op_types.matmul, lin, rin, op_dtype)
#         else:
#             # assume minimum dimension fits into both cores and max amount of dram queues
#             # for now...
#             min_dim = min(lin.shape[-2], min(lin.shape[-1], rin.shape[-1]))
#             row_fold = int(lin.shape[-2] / min_dim)
#             id_fold = int(lin.shape[-1] / min_dim)
#             col_fold = int(rin.shape[-1] / min_dim)
#             # fold, run, reduce, unfold
#             lin_folded, rin_folded = fold_inputs_for_mm(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
#             out = runtime.netlist.binary_tensor_op(tt_net_op_types.matmul, lin_folded, rin_folded, op_dtype)
#             print("Out shape: ",lin.shape, rin.shape, out.shape)
#             if(id_fold is not 1):
#                 import sys
#                 original_stdout = sys.stdout
#                 with open('reduce.txt', 'w') as f:
#                     sys.stdout = f # Change the standard output to the file we created.
#                     print('This message will be written to a file.')
#                     sys.stdout = original_stdout
#                 out = reduce(out, dim = -3, op_dtype = op_dtype, runtime=runtime)
#             else:
#                 # squeeze out the reduction dimension, if its just a placeholder
#                 out = out.squeeze(dim=-3)
#             out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

#         print("Fold factors: ", row_fold, col_fold, id_fold)
#     return out

def reduce(input, dim, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime=None):
    if(runtime == None):
        out = torch.sum(input, dim)
    else:
        # swap sum dimension and columns
        tens = input.swapaxes(-1,dim)
        # generated the ones tensor to reduce row-wise
        ones_shape = list(tens.shape)
        c = ones_shape.pop()
        r = ones_shape.pop()
        torch_ones_shape = ones_shape.copy()
        ones_shape.append(c)
        ones_shape.append(1)
        torch_ones_shape.append(c*input.virtual_block_size)
        torch_ones_shape.append(1*input.virtual_block_size)
        ones = tt_tensor(block_size=input.virtual_block_size, simd_cluster=runtime.simd_cluster, shape=ones_shape, dtype=op_dtype.dt)
        torch_ones = torch.ones(torch_ones_shape)
        ones.to_device(0, torch_ones)
        out = runtime.netlist.binary_tensor_op(tt_net_op_types.matmul, tens, ones, op_dtype)
        out = out.swapaxes(-1,dim)
        out = out.squeeze(dim)
    return out

def tt_binary_op(op: tt_net_op_types, lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(fold_factors is None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], min(lin.shape[-1], rin.shape[-1]))
        row_fold = int(lin.shape[-2] / min_dim)
        id_fold = int(lin.shape[-1] / min_dim)
        col_fold = int(rin.shape[-1] / min_dim)

        if(lin.shape[-2] < runtime.simd_cluster.r_cores and rin.shape[-1] < runtime.simd_cluster.c_cores and rin.shape[-2] < runtime.simd_cluster.queue_lim):
            row_fold = 1
            col_fold = 1
            id_fold = 1
            if(op.name == "matmul"):
                lin_folded, rin_folded = bcast_inputs_mm(lin,rin)
            else:
                lin_folded, rin_folded = bcast_inputs(lin,rin)
        else:
            if(op.name == "matmul"):
                lin_folded, rin_folded = fold_inputs_for_mm(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
            else:
                lin_folded, rin_folded = fold_inputs(lin, rin, row_fold, col_fold)
    else:
        row_fold, col_fold, id_fold = fold_factors
        if(row_fold is 1 and col_fold is 1 and id_fold is 1):
            if(op.name == "matmul"):
                lin_folded, rin_folded = bcast_inputs_mm(lin,rin)
            else:
                lin_folded, rin_folded = bcast_inputs(lin,rin)
        else:
            if(op.name == "matmul"):
                lin_folded, rin_folded = fold_inputs_for_mm(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
            else:
                lin_folded, rin_folded = fold_inputs(lin, rin, row_fold, col_fold)

    out = runtime.netlist.binary_tensor_op(op, lin_folded, rin_folded, op_dtype)
    print("Fold factors: ", op.name, row_fold, col_fold, id_fold, out.shape, lin.shape, rin.shape, lin_folded.shape, rin_folded.shape)

    if(id_fold is not 1):
        out = reduce(out, dim = -3, op_dtype = op_dtype, runtime=runtime)
    elif(row_fold is not 1 or col_fold is not 1):
        # squeeze out the reduction dimension, if its just a placeholder
        out = out.squeeze(dim=-3)

    if(row_fold is not 1 or col_fold is not 1 or id_fold is not 1):
        print("Fold factors: ", row_fold, col_fold, id_fold, out.shape)
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out

def tt_unary_elementwise_op(op: tt_net_op_types, lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(fold_factors is None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], lin.shape[-1])
        row_fold = int(lin.shape[-2] / min_dim)
        col_fold = int(lin.shape[-1] / min_dim)

        if(lin.shape[-2] < runtime.simd_cluster.r_cores and lin.shape[-1] < runtime.simd_cluster.c_cores):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)
    else:
        row_fold, col_fold = fold_factors
        if(row_fold is 1 and col_fold is 1):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)

    out = runtime.netlist.unary_tensor_op(op, lin_folded, op_dtype)

    if(row_fold is not 1 or col_fold is not 1):
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out


def matmul(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return torch.matmul(lin,rin)
    else:
        return tt_binary_op(tt_net_op_types.matmul, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def add(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin+rin
    else:
        return tt_binary_op(tt_net_op_types.add, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def subtract(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin - rin
    else:
        return tt_binary_op(tt_net_op_types.subtract, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def multiply(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin * rin
    else:
        return tt_binary_op(tt_net_op_types.multiply, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def exp(lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return torch.exp(lin)
    else:
        return tt_unary_elementwise_op(tt_net_op_types.exp, lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def linear(acts, weights, bias, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        out = torch.functional.linear(acts,weights,bias)
    else:
        out = matmul(acts, weights, op_dtype, runtime, fold_factors)
        out = add(out, bias, op_dtype, runtime, fold_factors)
    return out

def softmax(input, dim, output=None):
    out = functional.softmax(input, dim)
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


######
# Helper functions

def make_expand_mask(shape_list, dim, factor):
    expand_dim = shape_list[dim]
    out_list = shape_list
    for i in range(len(shape_list)):
        out_list[i] = -1
    out_list[dim] = expand_dim * factor
    return out_list

def fold_input(linput, rowfactor, colfactor):
    assert linput.shape[-1] % colfactor == 0
    assert linput.shape[-2] % rowfactor == 0

    # figure out reshaped dims
    out_cols = int(linput.shape[-1] / colfactor)
    out_rows = int(linput.shape[-2] / rowfactor)
    #reshape
    shape = list(linput.shape)
    # pop off the row and column dimensions
    shape.pop()
    shape.pop()
    # add the reshaped bottom dimensions
    shape.extend([rowfactor,out_rows,colfactor,out_cols])
    # reshape
    lrshp = linput.reshape(shape)
    # swap axes to make r,c at bottom
    lrshp = lrshp.swapaxes(-2,-3)
    return lrshp


def fold_inputs(linput, rinput, rowfactor, colfactor):
    assert linput.shape[-1] % colfactor == 0
    assert linput.shape[-2] % rowfactor == 0
    assert rinput.shape[-1] % colfactor == 0
    assert rinput.shape[-2] % rowfactor == 0

    # expand to equal dimensions before folding
    if(sum(list(linput.shape)) > sum(list(rinput.shape))):
        rrshp = rinput.broadcast_to(linput.shape)
    elif(sum(list(rinput.shape)) > sum(list(linput.shape))):
        lrshp = linput.broadcast_to(linput.shape)

    # figure out reshaped dims
    out_cols = int(linput.shape[-1] / colfactor)
    out_rows = int(linput.shape[-2] / rowfactor)
    #reshape
    shape = list(linput.shape)
    # pop off the row and column dimensions
    shape.pop()
    shape.pop()
    # add the reshaped bottom dimensions
    shape.extend([rowfactor,out_rows,colfactor,out_cols])
    # reshape
    lrshp = linput.reshape(shape)
    rrshp = rinput.reshape(shape)
    # swap axes to make r,c at bottom
    lrshp = lrshp.swapaxes(-2,-3)
    rrshp = rrshp.swapaxes(-2,-3)
    return lrshp, rrshp

######
# Input broadcasting will ensure that compatible inputs will
# be broadcast to the same dimensions
# this is done manually since the two chip dimensions throw
# off pytorch/numpy automatic broadcasting semantics
#
# *** For Non Matmuls
# find differences in dimensionality
# confirm dimensionality >= 3
# add dimensions right below chips to match dimensionality
# expand dimensions that differ
def bcast_inputs(lin, rin):
    assert len(lin.shape) >= 3
    assert len(rin.shape) >= 3
    dim_l = len(lin.shape)
    dim_r = len(rin.shape)
    if(dim_l > dim_r):
        dim_diff = dim_l - dim_r
        dim = dim_l
        for _ in range(dim_diff):
            lin = lin.unsqueeze(dim=2)
    elif(dim_r > dim_l):
        dim_diff = dim_r - dim_l
        dim = dim_r
        for _ in range(dim_diff):
            rin = rin.unsqueeze(dim=2)
    else:
        dim = dim_l
    expand_mask_l = []
    expand_mask_r = []
    # do not touch chip dims
    expand_mask_l.append(-1)
    expand_mask_l.append(-1)
    expand_mask_r.append(-1)
    expand_mask_r.append(-1)
    for i in range(2, dim): # do not touch chip dims
        if(lin.shape[i] > rin.shape[i]):
            assert lin.shape[i] % rin.shape[i] == 0
            expand_mask_r.append(int(lin.shape[i]/rin.shape[i]))
        else:
            expand_mask_r.append(-1)
        if(rin.shape[i] > lin.shape[i]):
            assert rin.shape[i] % lin.shape[i] == 0
            expand_mask_l.append(int(rin.shape[i]/lin.shape[i]))
        else:
            expand_mask_l.append(-1)
    return lin.expand(expand_mask_l), rin.expand(expand_mask_r)

######
# Input broadcasting will ensure that compatible inputs will
# be broadcast to the same dimensions
# this is done manually since the two chip dimensions throw
# off pytorch/numpy automatic broadcasting semantics
# *** For matmuls
# confirm inner dimension matches
# confirm dimensionality >= 4
# do not mess with bottom two dimensions
# for everything else - use method of non-matmuls
def bcast_inputs_mm(lin, rin):
    assert len(lin.shape) >= 4
    assert len(rin.shape) >= 4
    assert lin.shape[-1] == rin.shape[-2]
    dim_l = len(lin.shape)
    dim_r = len(rin.shape)
    if(dim_l > dim_r):
        dim_diff = dim_l - dim_r
        dim = dim_l
        for _ in range(dim_diff):
            lin = lin.unsqueeze(dim=2)
    elif(dim_r > dim_l):
        dim_diff = dim_r - dim_l
        dim = dim_r
        for _ in range(dim_diff):
            rin = rin.unsqueeze(dim=2)
    else:
        dim = dim_l
    expand_mask_l = []
    expand_mask_r = []
    # do not touch chip dims
    expand_mask_l.append(-1)
    expand_mask_l.append(-1)
    expand_mask_r.append(-1)
    expand_mask_r.append(-1)
    for i in range(2, dim-2): # do not touch chip dims OR row/col
        if(lin.shape[i] > rin.shape[i]):
            assert lin.shape[i] % rin.shape[i] == 0
            expand_mask_r.append(int(lin.shape[i]/rin.shape[i]))
        else:
            expand_mask_r.append(-1)
        if(rin.shape[i] > lin.shape[i]):
            assert rin.shape[i] % lin.shape[i] == 0
            expand_mask_l.append(int(rin.shape[i]/lin.shape[i]))
        else:
            expand_mask_l.append(-1)
    # do not touch row/col dims
    expand_mask_l.append(-1)
    expand_mask_l.append(-1)
    expand_mask_r.append(-1)
    expand_mask_r.append(-1)
    return lin.expand(expand_mask_l), rin.expand(expand_mask_r)

def fold_inputs_for_mm(linput, rinput, rowfactor, colfactor, idfactor):
    print(linput.shape)
    print(rinput.shape)
    # check basic assumptions
    assert linput.shape[-1] == rinput.shape[-2]
    assert linput.shape[-1] % idfactor == 0
    assert linput.shape[-2] % rowfactor == 0
    assert rinput.shape[-1] % colfactor == 0
    # expand to equal dimensions before folding
    lrshp, rrshp = bcast_inputs_mm(linput, rinput)
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
    print("Input shape: ",input.shape, rowfactor, colfactor)
    shape = list(input.shape)
    c = shape[-1]
    r = shape[-2]
    for _ in range(4):
        shape.pop()
    mc = c * colfactor
    mr = r * rowfactor
    shape.append(mr)
    shape.append(mc)
    print("Input shape: ",shape, rowfactor, colfactor)
    return input.swapaxes(-2,-3).reshape(shape)



