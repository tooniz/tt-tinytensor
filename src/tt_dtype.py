from enum import Enum

class tt_dtype(Enum):
    Bfp2_b = 1
    Bfp4_b = 2
    Bfp8_b = 3
    Float16 = 4
    Float16_b = 5
    Float32 = 6

def block_size_bytes(dtype, block_size, debug=True):
    tile_size_dict = {}
    if debug is False:
        tile_size_dict[tt_dtype.Bfp2_b] = 352
        tile_size_dict[tt_dtype.Bfp4_b] = 608
        tile_size_dict[tt_dtype.Bfp8_b] = 1120
        tile_size_dict[tt_dtype.Float16] = 2080
        tile_size_dict[tt_dtype.Float16_b] = 2080
        tile_size_dict[tt_dtype.Float32] = 4128
    else:
        tile_size_dict[tt_dtype.Bfp2_b] = 352 * 3
        tile_size_dict[tt_dtype.Bfp4_b] = 608 * 3
        tile_size_dict[tt_dtype.Bfp8_b] = 1120 * 3
        tile_size_dict[tt_dtype.Float16] = 2080 * 3
        tile_size_dict[tt_dtype.Float16_b] = 2080 * 3
        tile_size_dict[tt_dtype.Float32] = 4128 * 3

    return int((block_size/32)*(block_size/32)*tile_size_dict[dtype])

class tt_math_fidelity(Enum):
    LoFi = 1
    HiFi3 = 2

class tt_op_dtype():
    def __init__(self, dtype, dtype_intermed=tt_dtype.Float16, dtype_accum=tt_dtype.Float16, fidelity: tt_math_fidelity = tt_math_fidelity.HiFi3):
        self.dt = dtype
        self.dt_int = dtype_intermed
        self.dt_acc = dtype_accum
        self.fidelity = fidelity


