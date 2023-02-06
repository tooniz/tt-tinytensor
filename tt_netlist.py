import yaml
from enum import Enum

from tt_tensor import tt_tensor
from tt_simd_cluster import tt_simd_cluster
from tt_simd_cluster import tt_dtype
from tt_simd_cluster import tt_op_dtype
from tt_simd_cluster import tt_math_fidelity

class tt_net_op_types(Enum):
    matmul = 1
    add = 2
    subtract = 3
    multiply = 4
    queue = 5
    ram = 6

class tt_netlist:
    def __init__(self):
        self.doc = {}
        self.doc['queues'] = {}
        self.doc['devices'] = {}
        self.doc['graphs'] = {}
        self.queue_counter = 0
        self.op_counter = 0
        self.graph_counter = 0

    def start_netlist(self):
        pass
    def finish_netlist(self):
        pass
    def start_graph(self):
        pass
    def finish_graph(self):
        pass
    def add_queues(self, tensors: list, input = 'HOST'):
        # step through
        name = 'queue'
        for tensor in tensors:
            rqname = name + "_" + str(self.queue_counter)

            # go through the current r_c slice and define queue
            rdim = list(tensor.address_tensor.shape[-2])[0]
            cdim = list(tensor.address_tensor.shape[-1])[0]
            self.add_op(name = rqname, type = 'ram', block_size = tensor.block_size, grid_size = [rdim,cdim], inputs = [input], out_df = tensor.dtype, dram= tensor.get_dram_list(current_slice))
            self.queue_counter = self.queue_counter + 1

    def binary_tensor_op(self, op: tt_net_op_types, l_input: tt_tensor, r_input: tt_tensor, op_dtype: tt_op_dtype):
        # make output tensor
        out_tens = tt_tensor(block_size=l_input.virtual_block_size, simd_cluster=l_input.simd_cluster, shape=l_input.shape, dtype=op_dtype.dt)

        self.binary_slice_op(op, l_input, r_input, out_tens, op_dtype)

        self.dump_netlist()

        # compile dumped netlist
        # self.l_input.simd_cluster.compile_netlist()

        # run netlist
        # self.l_input.simd_cluster.run_netlist()

        # return output tiny tensor
        return out_tens

    def binary_slice_op(self, op: tt_net_op_types, l_input: tt_tensor, r_input: tt_tensor, output: tt_tensor, op_dtype: tt_op_dtype):
        # flatten out the dimensions that will be iterated through for computation
        l_input_flat = l_input.address_tensor.flatten(start_dim=2,end_dim=-3)
        r_input_flat = r_input.address_tensor.flatten(start_dim=2,end_dim=-3)
        output_flat = output.address_tensor.flatten(start_dim=2,end_dim=-3)

        iterations = l_input_flat.shape[2]

        op_name = op.name
        queue_name = 'queue'
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice)
            slice_op_name = op_name + "_" + str(slice)

            # go through the current r_c slice and define queue
            rdim = l_input.address_tensor.shape[-2]
            cdim = l_input.address_tensor.shape[-1]
            self.add_op(name = slice_queue_name+'_lin', type = tt_net_op_types.ram, block_size = l_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(l_input.dtype), dram= l_input.get_dram_list(slice))
            self.add_op(name = slice_queue_name+'_rin', type = tt_net_op_types.ram, block_size = r_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(r_input.dtype), dram= r_input.get_dram_list(slice))
            self.add_op(name = slice_queue_name+'_out', type = tt_net_op_types.ram, block_size = output.block_size, grid_size = [rdim,cdim], inputs = [slice_op_name], op_dtype = tt_op_dtype(output.dtype), dram= output.get_dram_list(slice))

        # make graphs and ops for current tensor slices
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice)
            slice_op_name = op_name + "_" + str(slice)
            rdim = output.address_tensor.shape[-2]
            cdim = output.address_tensor.shape[-1]

            self.add_op(name=slice_op_name, type=op, block_size=output.block_size, grid_size = [rdim,cdim], inputs = [slice_queue_name+'_lin', slice_queue_name+'_rin'], in_df = [l_input.dtype,r_input.dtype], op_dtype = op_dtype)

    def add_op(self, name: str, type: tt_net_op_types, \
            block_size: int, \
            grid_size: list, \
            inputs: list, # use inputs[0] as queue/ram input
            \
            op_dtype: tt_op_dtype,
            in_df: list = None,
            \
            entries: int = 1,
            target_device: int = 0,
            loc: str = 'dram',
            dram: list = [],
            \
            m_k: int = None, u_kt: int = None, \
            bias: str = None,
            relu_en: str = None, relu_mode: str = None, relu_threshold: float = None, \
            approximate_mode: str = None,
    ):
        # initialize stuff
        attributes = False
        op_val_dict = {}
        attributes_dict = {}

        # figure out mblock sizes
        grid_loc = [0,0]
        ublock_order = 'r' # r or c
        block_size_tiles = int(block_size / 32)
        ublock = [0,0]
        mblock = [block_size_tiles, block_size_tiles]

        # fill in required op/queue/ram descriptor fields
        op_val_dict['type'] = type.name
        op_val_dict['grid_loc'] = grid_loc
        op_val_dict['grid_size'] = grid_size
        op_val_dict['mblock'] = mblock
        op_val_dict['ublock'] = ublock
        op_val_dict['ublock_order'] = ublock_order
        op_val_dict['t'] = 1

        out_df = op_dtype.dt.name
        acc_df = op_dtype.dt_acc.name
        intermed_df = op_dtype.dt_int.name
        math_fidelity = op_dtype.fidelity.name

        if(type == tt_net_op_types.queue or type == tt_net_op_types.ram):
            op_val_dict['input'] = inputs[0]
            op_val_dict['entries'] = entries
            op_val_dict['target_device'] = target_device
            op_val_dict['loc'] = loc
            op_val_dict['dram'] = dram
            op_val_dict['df'] = out_df
        else:
            op_val_dict['inputs'] = inputs
            op_val_dict['in_df'] = list(map(lambda x:x.name,in_df)) #convert list of dtype objects to list of strings
            op_val_dict['out_df'] = out_df
            op_val_dict['intermed_df'] = intermed_df
            op_val_dict['acc_df'] = acc_df
            op_val_dict['math_fidelity'] = math_fidelity

        if(type == tt_net_op_types.matmul):
            attributes = True
            attributes_dict['m_k'] = m_k
            attributes_dict['u_kt'] = u_kt
        
        # Assemble attributes dictionary, if there are any
        if(bias == 'true'):
            attributes = True
            attributes_dict['bias'] = bias

        if(relu_en == 'true'):
            attributes = True
            attributes_dict['relu_en'] = 'true'
            attributes_dict['relu_mode'] = relu_mode
            attributes_dict['relu_threshold'] = relu_threshold

        if(approximate_mode == 'true'):
            attributes = True
            attributes_dict['approximate_mode'] = 'true'

        if(attributes is True):
            op_val_dict['attributes'] = attributes_dict

        # assign op val dictionary to the single entry op dict
        if(type == tt_net_op_types.queue or type == tt_net_op_types.ram):
            self.doc['queues'][name] = op_val_dict
        else:
            self.doc['graphs'][name] = op_val_dict

    def dump_netlist(self):
        pass
        #print(self.doc['queues'])
        #file = yaml.dump(self.doc, sort_keys = True, default_flow_style=True, width=1000)
        #print(file)
        with open('res.yaml', 'w') as yaml_file:
            yaml.dump(self.doc, yaml_file, default_flow_style=True)

def main():
    simd0  = tt_simd_cluster(2,2,(0,1,2,3))
    num_alloc_blocks = 100000
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,64,num_alloc_blocks,0)])
    simd0.set_up_allocators([(tt_dtype.Bfp8_b,128,num_alloc_blocks,0)])

    netlist0 = tt_netlist()

    block_size = 128
    dim_list0 = (1,2,3,8,8)
    dim_list1 = (1,2,3,4,4)
    dim_list2 = (1,2,3,2,2)

    bla0 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list0))
    bla1 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list1))
    bla2 = tt_tensor(block_size=block_size, simd_cluster=simd0, shape=tuple(dim_list2))

    out_tensor = netlist0.binary_tensor_op(tt_net_op_types.matmul, bla0, bla1, tt_op_dtype(tt_dtype.Bfp8_b))
    netlist0.dump_netlist()

    del(bla0)
    del(bla1)
    del(bla2)
    del(out_tensor)

if __name__ == "__main__":
    print("Testing TT netlist api")

    main()

    #main()
