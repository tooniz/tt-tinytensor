import torch
import logging
import torch.nn.functional as functional
from tt_tensor import tt_tensor
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
from tt_netlist import tt_netlist
from tt_netlist import tt_net_op_types
from eager_backend import BackendStatusCode

def reduce_id(input, dim, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime=None, fold_factors = None):
    if(runtime == None):
        out = torch.sum(input, dim)
    else:
        if(input.shape[dim] == 1):
            out = input.squeeze(dim)
        else:
            # swap sum dimension and columns
            tens = input.swapaxes(-1,dim)
            ones_shape = list(tens.shape)
            c = ones_shape.pop()
            r = ones_shape.pop()
            torch_ones_shape = ones_shape.copy()
            ones_shape.append(c)
            ones_shape.append(1)
            torch_ones_shape.append(c*input.virtual_block_size)
            torch_ones_shape.append(1*input.virtual_block_size)
            ones = tt_tensor(block_size=input.virtual_block_size, simd_cluster=runtime.simd_cluster, shape=ones_shape, dtype=op_dtype.dt)
            torch_zero_ones = reduce_ones(torch_ones_shape[-2],input.virtual_block_size)
            torch_zero_ones = torch_zero_ones.broadcast_to(torch_ones_shape)
            ones.to_device(0, torch_zero_ones)
            out = runtime.netlist.binary_tensor_op(tt_net_op_types.matmul, tens, ones, op_dtype)
            status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
            assert status == BackendStatusCode.Success
            runtime.backend.wait_for_idle()
            out = out.swapaxes(-1,dim)
            out = out.squeeze(dim)
    return out

def reduce(input, dim, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime=None, fold_factors = None):
    if(runtime == None):
        out = torch.sum(input, dim)
    else:
        if(input.shape[dim] == 1):
            out = input.squeeze(dim)
        else:
            # swap sum dimension and columns
            tens = input.swapaxes(-1,dim)
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
            out = matmul(tens, ones, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
            out = out.swapaxes(-1,dim)
            out = out.squeeze(dim)
    return out

def tt_binary_op(op: tt_net_op_types, lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    folded = False
    if(fold_factors == None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], min(lin.shape[-1], rin.shape[-1]))
        row_fold = int(lin.shape[-2] / min_dim)
        id_fold = int(lin.shape[-1] / min_dim)
        col_fold = int(rin.shape[-1] / min_dim)

        if(lin.shape[-2] <= runtime.simd_cluster.r_cores and rin.shape[-1] <= runtime.simd_cluster.c_cores and rin.shape[-2] < runtime.simd_cluster.queue_lim):
            row_fold = 1
            col_fold = 1
            id_fold = 1
            lin_folded, rin_folded = bcast_inputs_mm(lin,rin)
        else:
            lin_folded, rin_folded = fold_inputs_for_mm(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
            folded = True
    else:
        row_fold, col_fold, id_fold = fold_factors
        if((row_fold == 1) and (col_fold == 1) and (id_fold == 1)):
            lin_folded, rin_folded = bcast_inputs_mm(lin,rin)
        else:
            lin_folded, rin_folded = fold_inputs_for_mm(lin, rin, rowfactor=row_fold, colfactor=col_fold, idfactor=id_fold)
            folded = True
    out = runtime.netlist.binary_tensor_op(op, lin_folded, rin_folded, op_dtype)
    status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    runtime.backend.wait_for_idle()

    if(folded):
        if(id_fold != 1):
            out = reduce_id(out, dim = -3, op_dtype = op_dtype, runtime=runtime)
        else:
            # squeeze out the reduction dimension, if its just a placeholder
            out = out.squeeze(dim=-3)

        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out

def tt_binary_elementwise_op(op: tt_net_op_types, lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(fold_factors == None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], min(lin.shape[-1], rin.shape[-1]))
        row_fold = int(lin.shape[-2] / min_dim)
        col_fold = int(rin.shape[-1] / min_dim)

        if(lin.shape[-2] <= runtime.simd_cluster.r_cores and rin.shape[-1] <= runtime.simd_cluster.c_cores):
            row_fold = 1
            col_fold = 1
            lin_folded, rin_folded = bcast_inputs(lin,rin)
        else:
            lin_folded, rin_folded = fold_inputs(lin, rin, row_fold, col_fold)
    else:
        row_fold, col_fold, id_fold = fold_factors
        if((row_fold == 1) and (col_fold == 1)):
            lin_folded, rin_folded = bcast_inputs(lin,rin)
        else:
            lin_folded, rin_folded = fold_inputs(lin, rin, row_fold, col_fold)

    out = runtime.netlist.binary_tensor_op(op, lin_folded, rin_folded, op_dtype)
    status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    runtime.backend.wait_for_idle()

    if((row_fold != 1) or (col_fold != 1)):
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out

def tt_unary_elementwise_op(op: tt_net_op_types, lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(fold_factors is None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], lin.shape[-1])
        row_fold = int(lin.shape[-2] / min_dim)
        col_fold = int(lin.shape[-1] / min_dim)

        if(lin.shape[-2] <= runtime.simd_cluster.r_cores and lin.shape[-1] <= runtime.simd_cluster.c_cores):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)
    else:
        row_fold, col_fold, id_fold = fold_factors
        if((row_fold == 1) and (col_fold == 1)):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)
    out = runtime.netlist.unary_tensor_op(op, lin_folded, op_dtype)
    status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    runtime.backend.wait_for_idle()

    if((row_fold != 1) or (col_fold != 1)):
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out

def tt_reduce_op(op: tt_net_op_types, lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(fold_factors is None):
        # assume minimum dimension fits into both cores and max amount of dram queues
        # for now...
        min_dim = min(lin.shape[-2], lin.shape[-1])
        row_fold = int(lin.shape[-2] / min_dim)
        col_fold = int(lin.shape[-1] / min_dim)

        if(lin.shape[-2] <= runtime.simd_cluster.r_cores and lin.shape[-1] <= runtime.simd_cluster.c_cores):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)
    else:
        row_fold, col_fold, id_fold = fold_factors
        if((row_fold == 1) and (col_fold == 1)):
            lin_folded = lin
        else:
            lin_folded = fold_input(lin, row_fold, col_fold)

    assert col_fold == 1
    out = runtime.netlist.reduce_tensor_op(op, lin_folded, op_dtype)
    status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    runtime.backend.wait_for_idle()

    if((row_fold != 1) or (col_fold != 1)):
        out = unfold_output(out,rowfactor=row_fold,colfactor=col_fold)

    return out

def reduce_max(lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return torch.max(lin,dim=-1)
    else:
        return tt_reduce_op(tt_net_op_types.reduce, lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

# copy the given input tt_tensor to the given chips
def tt_broadcast_op(input, output, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None):
    runtime.netlist.unary_tensor_bcast_op(tt_net_op_types.nop, input, output, op_dtype)
    status = runtime.backend.compile_and_run_netlist(runtime.netlist.get_last_netlist_name(), {})
    assert status == BackendStatusCode.Success
    runtime.backend.wait_for_idle()

def broadcast(input, output, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None):
    if(runtime is None):
        pass # TODO: implement
    else:
        tt_broadcast_op(input, output, op_dtype, runtime)

def matmul(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return torch.matmul(lin,rin)
    else:
        return tt_binary_op(tt_net_op_types.matmul, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def add(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin+rin
    else:
        return tt_binary_elementwise_op(tt_net_op_types.add, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def subtract(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin - rin
    else:
        return tt_binary_elementwise_op(tt_net_op_types.subtract, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def multiply(lin, rin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return lin * rin
    else:
        return tt_binary_elementwise_op(tt_net_op_types.multiply, lin, rin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def exp(lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return torch.exp(lin)
    else:
        return tt_unary_elementwise_op(tt_net_op_types.exp, lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def reciprocal(lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return (1/lin)
    else:
        return tt_unary_elementwise_op(tt_net_op_types.reciprocal, lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def sqrt(lin, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        return (1/lin)
    else:
        return tt_unary_elementwise_op(tt_net_op_types.sqrt, lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)

def linear(acts, weights, bias, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        out = torch.functional.linear(acts,weights,bias)
    else:
        out = matmul(acts, weights, op_dtype, runtime, fold_factors)
        out = add(out, bias, op_dtype, runtime, fold_factors)
    return out

def softmax(lin, dim, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if(runtime is None):
        out = functional.softmax(input, dim)
    else:
        #### Everything here is fairly straightforward except the fold factor gymnastics
        # The reduce matmul has a fixed column width, so can't be folding columns there
        # the reduced tensor going into reciprocal is one wide so can't be folding columns there
        #
        # For the exp() and multiply() can fold both rows and columns, the same way - that makes sense
        # for the reduce matmul can fold rows (just like exp and multiply, makes sense)
        # and can fold inner dimension - the only op that takes the inner dim folding argument is matmul
        # again, makes sense
        expo = exp(lin=lin, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
        reduce_fold_factors = list(fold_factors)
        reduce_fold_factors[1] = 1 # We can't fold columns on the reduction, or the reduced tensor going into reciprocal
        red  = reduce(expo, dim, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
        red = red.unsqueeze(-1)
        red_recip = reciprocal(red, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
        print(expo.shape, red.shape, red_recip.shape, expo.dtype.name, red_recip.dtype.name)
        out = multiply(expo, red_recip, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    return out

def layer_norm(lin, beta, gamma, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    assert runtime is not None, "Don't call tt_functional op without runtime"

    import pdb
    #pdb.set_trace()
    # MEAN
    reduce_fold_factors = list(fold_factors)
    reduce_fold_factors[1] = 1 # We can't fold columns on the reduction, or the reduced tensor going into reciprocal
    sumo = reduce(lin, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
    sumo = sumo.unsqueeze(-1)
    numel = torch.fill(torch.empty((1,1,1,lin.block_size, lin.block_size)), 1/(lin.shape[-1]*lin.block_size)) # tensor filled w/ number of elements in reduce dim
    tt_numel = tt_tensor(lin.block_size, runtime.simd_cluster, torch_tensor=numel, dtype=lin.dtype).to_device(0, numel)
    avg = multiply(sumo, tt_numel, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    # VARIANCE
    diff = subtract(lin, avg, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    sqr = multiply(diff, diff, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    var_unnormalized = reduce(sqr, dim=-1, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
    # TODO: divide by numel to compute variance
    var = multiply(var_unnormalized, tt_numel, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    sqrto = sqrt(var, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
    recip = reciprocal(sqrto, op_dtype=op_dtype, runtime=runtime, fold_factors=tuple(reduce_fold_factors))
    recip = recip.unsqueeze(-1)
    scaled_diff = multiply(diff, recip, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    g_scaled = multiply(scaled_diff, gamma, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    out = add(g_scaled, beta, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    return out

def gelu(input, op_dtype = tt_op_dtype(tt_dtype.Float16), runtime = None, fold_factors: tuple = None):
    if runtime is None:
        out = functional.gelu(input)
    else:
        out = tt_unary_elementwise_op(tt_net_op_types.gelu, input, op_dtype=op_dtype, runtime=runtime, fold_factors=fold_factors)
    return out

def relu(input, output=None):
    out = functional.relu(input)
    if(output != None):
        output.copy_(out)
    return out


######
# Helper functions
def reduce_ones(r, block_size):
    first_one_offset = 0
    out_tensor = torch.zeros(r,block_size)
    for col in range(block_size):
        period_cntr = 0
        for row in range(r):
            if(row >= first_one_offset):
                if(period_cntr % block_size == 0):
                    out_tensor[row][col] = 1.0
                period_cntr = period_cntr + 1
        first_one_offset = first_one_offset + 1
    return out_tensor

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
    # expand to equal dimensions before folding
    lrshp, rrshp = bcast_inputs(linput,rinput)

    assert lrshp.shape[-1] % colfactor == 0
    assert lrshp.shape[-2] % rowfactor == 0
    assert rrshp.shape[-1] % colfactor == 0
    assert rrshp.shape[-2] % rowfactor == 0

    # figure out reshaped dims
    out_cols = int(lrshp.shape[-1] / colfactor)
    out_rows = int(lrshp.shape[-2] / rowfactor)
    #reshape
    shape = list(lrshp.shape)
    # pop off the row and column dimensions
    shape.pop()
    shape.pop()
    # add the reshaped bottom dimensions
    shape.extend([rowfactor,out_rows,colfactor,out_cols])
    # reshape
    lrshp = lrshp.reshape(shape)
    rrshp = rrshp.reshape(shape)
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
            rin = rin.unsqueeze(dim=2)
    elif(dim_r > dim_l):
        dim_diff = dim_r - dim_l
        dim = dim_r
        for _ in range(dim_diff):
            lin = lin.unsqueeze(dim=2)
    else:
        dim = dim_l
    expand_mask_l = []
    expand_mask_r = []
    for i in range(0, dim): # do not touch chip dims
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
    assert len(lin.shape) >= 2
    assert len(rin.shape) >= 2
    assert lin.shape[-1] == rin.shape[-2]
    dim_l = len(lin.shape)
    dim_r = len(rin.shape)
    if(dim_l > dim_r):
        dim_diff = dim_l - dim_r
        dim = dim_l
        for _ in range(dim_diff):
            rin = rin.unsqueeze(dim=0)
    elif(dim_r > dim_l):
        dim_diff = dim_r - dim_l
        dim = dim_r
        for _ in range(dim_diff):
            lin = lin.unsqueeze(dim=0)
    else:
        dim = dim_l
    assert len(lin.shape) == len(rin.shape)
    expand_mask_l = []
    expand_mask_r = []

    for i in range(0, dim-2): # do not touch row/col dims
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
    # check basic assumptions
    assert linput.shape[-1] == rinput.shape[-2]
    assert linput.shape[-1] % idfactor == 0
    assert linput.shape[-2] % rowfactor == 0
    assert rinput.shape[-1] % colfactor == 0
    # expand to equal dimensions before folding
    lrshp, rrshp = bcast_inputs_mm(linput, rinput)
    # figure out reshaped dims
    out_cols = int(rrshp.shape[-1] / colfactor)
    out_rows = int(lrshp.shape[-2] / rowfactor)
    out_id = int(lrshp.shape[-1] / idfactor)
    #reshape
    l_shape = list(lrshp.shape)
    r_shape = list(rrshp.shape)
    # pop off the row and column dimensions
    l_shape.pop()
    l_shape.pop()
    r_shape.pop()
    r_shape.pop()
    # add the reshaped bottom dimensions
    l_shape.extend([rowfactor,out_rows,idfactor,out_id])
    r_shape.extend([idfactor,out_id,colfactor,out_cols])

    # reshape
    lrshp = lrshp.reshape(l_shape)
    rrshp = rrshp.reshape(r_shape)
    # swap axes to make r,c at bottom
    lrshp = lrshp.swapaxes(-2,-3) # rowfactor,  idfactor, out_rows, out_id
    rrshp = rrshp.swapaxes(-2,-3) # idfactor , colfactor, out_id  , out_cols 
    rrshp = rrshp.swapaxes(-3,-4) # colfactor,  idfactor, out_id  , out_cols
    # add extra dims for broadcasting to each others shape
    lrshp = lrshp.unsqueeze(-4) # rowfactor,1,idfactor,out_rows,out_id
    rrshp = rrshp.unsqueeze(-5) # 1,colfactor,idfactor,out_id,out_cols
    # broadcast to each others shapes
    lrshp, rrshp = bcast_inputs_mm(lrshp, rrshp)
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



