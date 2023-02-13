import random
import math
import torch
import tt_functional as tt_functional
import torch.nn.functional as torch_functional

def tt_ffn(input, w0, b0, w1, b1):
    # derive tensor shapes
    in_shape = input.shape
    ff2_shape = in_shape
    ff1_shape = in_shape[:-1] + (w0.shape[-1],)
    # declare tensors before use
    ff1_t = torch.Tensor(ff1_shape)
    gelu_t = torch.Tensor(ff1_shape)
    ff2_t = torch.Tensor(ff2_shape)
    out_t = torch.Tensor(ff2_shape)
    # run functional ops
    tt_functional.linear(input,w0,b0,ff1_t)
    tt_functional.gelu(ff1_t,gelu_t)
    tt_functional.linear(gelu_t,w1,b1,ff2_t)
    tt_functional.add(input,ff2_t, out_t)
    return out_t

def torch_ffn(input, w0, b0, w1, b1):
    x = torch_functional.linear(input,w0,b0)
    x = torch_functional.gelu(x)
    x = torch_functional.linear(x,w1,b1)
    x = x + input
    return x

def transpose_for_scores(x: torch.Tensor, d_model, num_heads) -> torch.Tensor:
    new_x_shape = x.size()[:-1] + (num_heads, int(d_model/num_heads))
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def torch_selfattn(d_model, num_heads, input, wk, bk, wq, bq, wv, bv, wl, bl):
    # qkv projections
    q = torch_functional.linear(input,wq,bq)
    k = torch_functional.linear(input,wk,bk)
    v = torch_functional.linear(input,wv,bv)
 
    # reshape so that heads are 2D slices
    # and num_heads is the 3rd dimensions
    q_s = transpose_for_scores(q, d_model, num_heads)
    k_s = transpose_for_scores(k, d_model, num_heads)
    v_s = transpose_for_scores(v, d_model, num_heads)

    # transpose key heads for attn score matmul
    k_t = k_s.transpose(-1, -2)

    # attn score matmul
    attn = torch.matmul(q_s, k_t)

    # normalize by 1/sqrt(d_model)
    attn_norm = attn / math.sqrt(d_model)

    # softmax
    attn_smax = torch_functional.softmax(attn_norm, dim=-1)

    # linearly recombine v_s
    v_mix = torch.matmul(attn_smax, v_s)

    # reshape v_mix back to 2D (heads stacked)
    v_mix = v_mix.permute(0, 2, 1, 3).contiguous()
    new_v_mix_shape = v_mix.size()[:-2] + (d_model,)
    v_mix = v_mix.view(new_v_mix_shape)

    # apply final linear
    lin = torch_functional.linear(v_mix,wl,bl)
    return lin


####
# Below here is tt_functional test code
def test_ffn():
    # randomize dims and num heads
    dmodel_mul = random.randint(1,32)
    seqlen_mul = random.randint(1,32)
    dmodel = 32 * dmodel_mul
    seqlen = 32 * seqlen_mul

    # generate random weights
    input = torch.randn(seqlen,dmodel)
    w0 = torch.randn(dmodel,dmodel)
    w1 = torch.randn(dmodel,dmodel)
    b0 = torch.randn(1,dmodel)
    b1 = torch.randn(1,dmodel)

    # generate random input tensors
    input = torch.randn(seqlen,dmodel)

    tt_out = tt_ffn(input,w0,b0,w1,b1)
    torch_out = torch_ffn(input,w0,b0,w1,b1)

    assert torch.allclose(tt_out,torch_out)
    print("Passed: tt ffn vs torch ffn")

def test_self_attn():
    # randomize dims
    dmodel_mul = random.randint(1,512)
    seqlen_mul = random.randint(1,512)
    num_head_pow = random.randint(1,3)

    dmodel = 32 * dmodel_mul
    seqlen = 32 * seqlen_mul
    num_heads = 2 ** num_head_pow

    # generate random weights
    wk = torch.randn(dmodel,dmodel)
    bk = torch.randn(1,dmodel)
    wq = torch.randn(dmodel,dmodel)
    bq = torch.randn(1,dmodel)
    wv = torch.randn(dmodel,dmodel)
    bv = torch.randn(1,dmodel)
    wl = torch.randn(dmodel,dmodel)
    bl = torch.randn(1,dmodel)

    # generate random input tensors
    input = torch.randn(1,seqlen,dmodel)

    # tt_out = tt_ffn(input,w0,b0,w1,b1)
    torch_out = torch_selfattn(dmodel, num_heads, input,wk,bk,wq,bq,wv,bv,wl,bl)

def main():
    print("Testing TT functional!")
    #test_self_attn()
    for x in range(10):
        test_ffn()

if __name__ == "__main__":
    main()