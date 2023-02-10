import yaml
from enum import Enum

from tt_tensor import tt_tensor
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
from tt_dtype import tt_math_fidelity

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

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
            self.add_op(lin_tensor=l_input, name = slice_queue_name+'_lin', type = tt_net_op_types.ram, block_size = l_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(l_input.dtype), dram= l_input.get_dram_list(slice))
            self.add_op(lin_tensor=l_input, name = slice_queue_name+'_rin', type = tt_net_op_types.ram, block_size = r_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(r_input.dtype), dram= r_input.get_dram_list(slice))
            self.add_op(lin_tensor=l_input, name = slice_queue_name+'_out', type = tt_net_op_types.ram, block_size = output.block_size, grid_size = [rdim,cdim], inputs = [slice_op_name], op_dtype = tt_op_dtype(output.dtype), dram= output.get_dram_list(slice))

        # make graphs and ops for current tensor slices
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice)
            slice_op_name = op_name + "_" + str(slice)
            rdim = output.address_tensor.shape[-2]
            cdim = output.address_tensor.shape[-1]

            self.add_op(lin_tensor=l_input, name=slice_op_name, type=op, block_size=output.block_size, grid_size = [rdim,cdim], inputs = [slice_queue_name+'_lin', slice_queue_name+'_rin'], in_df = [l_input.dtype,r_input.dtype], op_dtype = op_dtype)

    def add_op(self, lin_tensor: tt_tensor, name: str, type: tt_net_op_types, \
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
            bias: str = None,
            relu_en: str = None, relu_mode: str = None, relu_threshold: float = None, \
            approximate_mode: str = None,
    ):
        # initialize stuff
        attributes = False
        op_val_dict = {}
        attributes_dict = {}

        grid_loc = [0,0]
        ublock_order = 'r' # r or c

        #
        # figure out mbloc/ublock sizes
        # and m_k & u_kt
        #
        block_size_tiles = int(block_size / 32)
        if(block_size > 256):
            ublock_size_tiles = int(block_size_tiles / 4)
        elif(block_size == 128):
            ublock_size_tiles = int(block_size_tiles / 2)
        elif(block_size < 128):
            ublock_size_tiles = block_size_tiles
        ublock = [ublock_size_tiles,ublock_size_tiles]
        mblock = [int(block_size_tiles/ublock_size_tiles), int(block_size_tiles/ublock_size_tiles)]
        inner_dim_tiles = int((lin_tensor.address_tensor.shape[-1] * lin_tensor.block_size) / 32)
        u_kt = 2
        m_k = int(inner_dim_tiles / u_kt)

        # fill in required op/queue/ram descriptor fields
        op_val_dict['type'] = type.name
        if(type is not tt_net_op_types.queue and type is not tt_net_op_types.ram):
            op_val_dict['grid_loc'] = grid_loc
        op_val_dict['grid_size'] = grid_size
        op_val_dict['mblock'] = mblock
        op_val_dict['ublock'] = ublock
        if(type is not tt_net_op_types.queue and type is not tt_net_op_types.ram):
            op_val_dict['ublock_order'] = ublock_order
            op_val_dict['buf_size_mb'] = 2
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
            op_val_dict['untilize_output'] = False

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
            self.doc['graphs']['graph_op0'] = {}
            self.doc['graphs']['graph_op0']['target_device'] = 0
            self.doc['graphs']['graph_op0']['input_count'] = 4
            self.doc['graphs']['graph_op0'][name] = op_val_dict

    def dump_netlist(self):
        self.doc['devices'] = {}
        self.doc['devices']['arch'] = 'grayskull'
        with open('res.yaml', 'w') as yaml_file:
            yaml.dump(self.doc, yaml_file, Dumper=IndentDumper, sort_keys=False, default_flow_style=False)
            yaml_file.close()
    
        end_string = """programs:
- op_matmul:
    - var: {$p_microbatch_count: 1}
    - var: {$c_microbatch_size: 1, $c_one: 1, $c_zero: 0}
    - staticvar: {$gptr_q0: 0, $lptr_q0: 0}
    - loop: $p_microbatch_count
    -   execute: {graph_name: graph_op0, queue_settings: {
        queue_0_lin: {prologue: false, epilogue: false, zero: false, rd_ptr_global: $c_zero, wr_ptr_global: $c_zero},
        queue_0_rin: {prologue: false, epilogue: false, zero: false, rd_ptr_global: $c_zero, wr_ptr_global: $c_zero}}}
    - endloop
test-config:
    comparison-config:
        type: AllCloseHw
        atol: 0.01
        rtol: 0.15
        check_pct: 0.75
        check_pcc: 0.97
        verbosity: Concise
    stimulus-config:
        type: Normal
        normal_mean: 0.0
        normal_stddev: 0.25"""
        file_object = open('res.yaml', 'a')
        file_object.write(end_string)
        file_object.close()

