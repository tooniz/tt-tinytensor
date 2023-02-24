import yaml
from enum import Enum

from tt_tensor import tt_tensor
from tt_dtype import tt_dtype
from tt_dtype import tt_op_dtype
from tt_dtype import tt_math_fidelity
from eager_backend import BackendDevice
import IPython
from typing import List, Set, Dict, Tuple

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
    nop = 10
    reciprocal = 11
    exp = 12
    gelu = 13
    sqrt = 14
    reduce = 15


class tt_netlist:
    def __init__(self, arch=BackendDevice.Grayskull):
        self.doc = {}
        self.doc['queues'] = {}
        self.doc['devices'] = {}
        self.doc['graphs'] = {}
        self.queue_counter = 0
        self.op_counter = 0
        self.graph_counter = 0

        self.last_netlist_filename = None
        self.next_netlist_idx = 0
        self.arch = arch

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

    def unary_tensor_op(self, op: tt_net_op_types, l_input: tt_tensor, op_dtype: tt_op_dtype):
        # make output tensor
        out_tens = tt_tensor(block_size=l_input.virtual_block_size, simd_cluster=l_input.simd_cluster, shape=l_input.shape, dtype=op_dtype.dt)
        self.unary_slice_op(op, l_input, out_tens, op_dtype)
        self.dump_netlist()
        return out_tens

    def unary_slice_op(self, op: tt_net_op_types, l_input: tt_tensor, output: tt_tensor, op_dtype: tt_op_dtype):
        # flatten out the dimensions that will be iterated through for computation
        l_input_flat = l_input.address_tensor.flatten(start_dim=2,end_dim=-3)
        output_flat = output.address_tensor.flatten(start_dim=2,end_dim=-3)

        iterations = l_input_flat.shape[2]

        op_name = op.name
        queue_name = 'queue'
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            slice_op_name = op_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)

            # go through the current r_c slice and define queue
            rdim = l_input.address_tensor.shape[-2]
            cdim = l_input.address_tensor.shape[-1]
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_queue_name+'_lin', type = tt_net_op_types.ram, block_size = l_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(l_input.dtype), dram= l_input.get_dram_list(slice))
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_queue_name+'_out', type = tt_net_op_types.ram, block_size = output.block_size, grid_size = [rdim,cdim], inputs = [slice_op_name], op_dtype = tt_op_dtype(output.dtype), dram= output.get_dram_list(slice))

        # make graphs and ops for current tensor slices
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            slice_op_name = op_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            rdim = output.address_tensor.shape[-2]
            cdim = output.address_tensor.shape[-1]

            self.add_op(slice_idx=slice, lin_tensor=l_input, name=slice_op_name, type=op, block_size=output.block_size, grid_size = [rdim,cdim], inputs = [slice_queue_name+'_lin'], in_df = [l_input.dtype], op_dtype = op_dtype)

    def reduce_tensor_op(self, op: tt_net_op_types, l_input: tt_tensor, op_dtype: tt_op_dtype):
        lshape = list(l_input.shape)
        lshape.pop()
        lshape.append(1)

        out_tens = tt_tensor(block_size=l_input.virtual_block_size, simd_cluster=l_input.simd_cluster, shape=tuple(lshape), dtype=op_dtype.dt)

        self.unary_slice_op(tt_net_op_types.reduce, l_input, out_tens, op_dtype)

        self.dump_netlist()

    def unary_slice_bcast_op(self, op: tt_net_op_types, l_input: tt_tensor, output: tt_tensor, op_dtype: tt_op_dtype):
        # flatten out the dimensions that will be iterated through for computation
        l_input_flat = l_input.address_tensor.flatten(start_dim=2,end_dim=-3)

        iterations = l_input_flat.shape[2]

        op_name = op.name
        queue_name = 'queue'
        for slice in range(iterations):
            # define input_queue
            slice_input_queue_name = queue_name + "_" + str(slice) + "_" + str(self.next_netlist_idx) + "_lin"
            rdim = l_input.address_tensor.shape[-2]
            cdim = l_input.address_tensor.shape[-1]
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_input_queue_name, type = tt_net_op_types.ram, block_size = l_input.block_size, grid_size = [rdim,cdim], inputs = ['HOST'], op_dtype = tt_op_dtype(l_input.dtype), dram= l_input.get_dram_list(slice))

            dim_chip_r = output.shape[0]
            dim_chip_c = output.shape[1]
            # for each chip, define output_queue and and a nop to forward the data
            for chip_r in range(dim_chip_r):
                for chip_c in range(dim_chip_c):
                    # TODO: convert logical ids to physical ids
                    chip_id = chip_r * dim_chip_r + chip_c
                    slice_op_name = op_name + "_" + str(chip_id) + "_" + str(slice) + "_" + str(self.next_netlist_idx)
                    slice_output_queue_name = queue_name + "_" + str(chip_id) + "_" + str(slice) + "_" + str(self.next_netlist_idx) + "_out"
                    self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_output_queue_name, type = tt_net_op_types.ram, block_size = output.block_size, grid_size = [rdim,cdim], inputs = [slice_op_name], op_dtype = tt_op_dtype(output.dtype), dram= output.get_dram_list(slice), target_device=chip_id)
                    rdim = output.address_tensor.shape[-2]
                    cdim = output.address_tensor.shape[-1]
                    self.add_op(slice_idx=slice, lin_tensor=l_input, name=slice_op_name, type=op, block_size=output.block_size, grid_size = [rdim,cdim], inputs = [slice_input_queue_name], in_df = [l_input.dtype], op_dtype = op_dtype, target_device=chip_id)
                    slice_input_queue_name = slice_op_name

    def unary_tensor_bcast_op(self, op: tt_net_op_types, l_input: tt_tensor, output: tt_tensor, op_dtype: tt_op_dtype):
        self.unary_slice_bcast_op(tt_net_op_types.nop, l_input, output, op_dtype)
        self.dump_netlist()

    def binary_tensor_op(self, op: tt_net_op_types, l_input: tt_tensor, r_input: tt_tensor, op_dtype: tt_op_dtype):
        # make output tensor
        if(op is tt_net_op_types.matmul):
            lshape = list(l_input.shape)
            r_shape = list(r_input.shape)
            lshape.pop()
            lshape.append(r_shape[-1])
            out_tens = tt_tensor(block_size=l_input.virtual_block_size, simd_cluster=l_input.simd_cluster, shape=tuple(lshape), dtype=op_dtype.dt)
        else:
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
            slice_queue_name = queue_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            slice_op_name = op_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)

            # go through the current r_c slice and define queue
            if(l_input.transpose_r_c):
                rdim_l = l_input.address_tensor.shape[-1]
                cdim_l = l_input.address_tensor.shape[-2]
                rdim_out = l_input.address_tensor.shape[-1]
            else:
                rdim_l = l_input.address_tensor.shape[-2]
                cdim_l = l_input.address_tensor.shape[-1]
                rdim_out = l_input.address_tensor.shape[-2]

            if(r_input.transpose_r_c):
                rdim_r = r_input.address_tensor.shape[-1]
                cdim_r = r_input.address_tensor.shape[-2]
                cdim_out = r_input.address_tensor.shape[-1]
            else:
                rdim_r = r_input.address_tensor.shape[-2]
                cdim_r = r_input.address_tensor.shape[-1]
                cdim_out = r_input.address_tensor.shape[-1]
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_queue_name+'_lin', type = tt_net_op_types.ram, block_size = l_input.block_size, grid_size = [rdim_l,cdim_l], inputs = ['HOST'], op_dtype = tt_op_dtype(l_input.dtype), dram= l_input.get_dram_list(slice))
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_queue_name+'_rin', type = tt_net_op_types.ram, block_size = r_input.block_size, grid_size = [rdim_r,cdim_r], inputs = ['HOST'], op_dtype = tt_op_dtype(r_input.dtype), dram= r_input.get_dram_list(slice))
            self.add_op(slice_idx=slice, lin_tensor=l_input, name = slice_queue_name+'_out', type = tt_net_op_types.ram, block_size = output.block_size, grid_size = [rdim_out,cdim_out], inputs = [slice_op_name], op_dtype = tt_op_dtype(output.dtype), dram= output.get_dram_list(slice))

        # make graphs and ops for current tensor slices
        for slice in range(iterations):
            # make rams/queues for current tensor slices
            slice_queue_name = queue_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            slice_op_name = op_name + "_" + str(slice) + "_" + str(self.next_netlist_idx)
            rdim = output.address_tensor.shape[-2]
            cdim = output.address_tensor.shape[-1]

            self.add_op(slice_idx=slice, lin_tensor=l_input, name=slice_op_name, type=op, block_size=output.block_size, grid_size = [rdim,cdim], inputs = [slice_queue_name+'_lin', slice_queue_name+'_rin'], in_df = [l_input.dtype,r_input.dtype], op_dtype = op_dtype, lin_transpose=l_input.transpose_r_c, rin_transpose=r_input.transpose_r_c)

    def add_op(self, slice_idx: int, lin_tensor: tt_tensor, name: str, type: tt_net_op_types, \
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
            \
            lin_transpose = False,
            rin_transpose = False,
            approximate_mode: str = None,
    ):
        # initialize stuff
        attributes = False
        op_val_dict = {}
        attributes_dict = {}
        lin_tm_list = []
        rin_tm_list = []

        grid_loc = [0,0]
        ublock_order = 'r' #'r' # r or c

        #
        # figure out mbloc/ublock sizes
        # and m_k & u_kt
        #
        block_size_tiles = int(block_size / 32)
        ublock_size_tiles = 2
        # if(block_size > 256):
        #     ublock_size_tiles = int(block_size_tiles / 4)
        # elif(block_size == 128):
        #     ublock_size_tiles = int(block_size_tiles / 2)
        # elif(block_size < 128):
        #     ublock_size_tiles = block_size_tiles
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

        if(type == tt_net_op_types.reduce):
            attributes = True
            attributes_dict['dim'] = 'c'
            attributes_dict['type'] = 'max'
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

        if(lin_transpose is True):
            lin_tm_list.append('transpose')
            op_val_dict['input_0_tms'] = lin_tm_list

        if(rin_transpose is True):
            rin_tm_list.append('transpose')
            op_val_dict['input_1_tms'] = rin_tm_list

        # assign op val dictionary to the single entry op dict
        if(type == tt_net_op_types.queue or type == tt_net_op_types.ram):
            self.doc['queues'][name] = op_val_dict
        else:
            self.graph_op_name = 'graph_op_' + str(target_device) + "_" + str(slice_idx) + "_" + str(self.next_netlist_idx)
            self.doc['graphs'][self.graph_op_name] = {}
            self.doc['graphs'][self.graph_op_name]['target_device'] = target_device
            self.doc['graphs'][self.graph_op_name]['input_count'] = 1
            self.doc['graphs'][self.graph_op_name][name] = op_val_dict

    def get_last_netlist_name(self):
        assert self.last_netlist_filename != None
        return self.last_netlist_filename

    def arch_to_str(self, arch):
        arch_map = {BackendDevice.Grayskull: 'grayskull',
                    BackendDevice.Wormhole: 'wormhole'}
        return arch_map[arch]

    def dump_netlist(self):
        self.last_netlist_filename = 'netlist_' + str(self.next_netlist_idx) + '.yaml'
        self.doc['devices'] = {}
        self.doc['devices']['arch'] = self.arch_to_str(self.arch)
        with open(self.last_netlist_filename, 'w') as yaml_file:
            yaml.dump(self.doc, yaml_file, Dumper=IndentDumper, sort_keys=False, default_flow_style=False)
            yaml_file.close()
    
        program_string = "programs:\n"
        program_name_string = "  - op_" + str(self.next_netlist_idx) + ":\n"
        var_string = "    - var: {$c_zero: 0}\n"
        file_object = open(self.last_netlist_filename, 'a')
        file_object.write(program_string)
        file_object.write(program_name_string)
        file_object.write(var_string)

        for graph in self.doc['graphs'].items():
            key,value = graph
            exec_string = "    - execute: {graph_name: " + key + ", queue_settings: {}}\n"
            file_object.write(exec_string)

        file_object.close()
        self.next_netlist_idx = self.next_netlist_idx + 1

        self.doc = {}
        self.doc['queues'] = {}
        self.doc['devices'] = {}
        self.doc['graphs'] = {}
        self.queue_counter = 0
        self.op_counter = 0
        self.graph_counter = 0