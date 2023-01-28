from tt_tensor import tt_tensor
from tt_tensor import tt_tensor
from tt_simd_cluster import tt_simd_cluster
from tt_malloc import tt_malloc
import yaml

tt_net_op_types  = {"matmul": 'matmul','add': 'add','subtract': 'subtract','multiply': 'multiply','queue': 'queue','ram': 'ram'}
tt_data_formats  = {"Bfp2_b": 'Bfp2_b','Bfp4_b': 'Bfp4_b','Bfp8_b': 'Bfp8_b','Float16_b': 'Float16_b','Float32': 'Float32','Float16_a': 'Float16_a'}
tt_math_fidelity = {"LoFi": 'LoFi'}

class tt_netlist:
    def __init__(self):
        self.doc = {}
        self.doc['queues'] = {}
        self.doc['devices'] = {}
        self.doc['graphs'] = {}

    def start_netlist(self):
        pass
    def finish_netlist(self):
        pass
    def start_graph(self):
        pass
    def finish_graph(self):
        pass
    def add_queues(self, name: str, tensors: list, current_slice: tuple, input = 'HOST'):
        # step through
        counter = 0
        for tensor in tensors:
            rqname = name + "_" + str(counter)
            print("Ljabesi: ", tensor.address_tensor.shape,current_slice)
            # go through the current r_c slice and define queue
            rdim = list(tensor.address_tensor[current_slice].shape)[0]
            cdim = list(tensor.address_tensor[current_slice].shape)[1]
            self.add_op(name = rqname, type = 'ram', block_size = tensor.block_size, grid_size = [rdim,cdim], inputs = [input], out_df = tensor.data_format, dram= tensor.address_tensor[current_slice].flatten().tolist())
            counter = counter + 1

    def add_op(self, name: str, type: str, \
            block_size: int, \
            grid_size: list, \
            inputs: list, # use inputs[0] as queue/ram input
            out_df: str, # out_df must be supplied for all
            \
            in_df: list = None, intermed_df: str = None, acc_df: str = None, math_fidelity: str = None,
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
        op_val_dict['type'] = type
        op_val_dict['grid_loc'] = grid_loc
        op_val_dict['grid_size'] = grid_size
        op_val_dict['mblock'] = mblock
        op_val_dict['ublock'] = ublock
        op_val_dict['ublock_order'] = ublock_order
        op_val_dict['t'] = 1

        if(type == 'queue' or type == 'ram'):
            op_val_dict['input'] = inputs[0]
            op_val_dict['entries'] = entries
            op_val_dict['target_device'] = target_device
            op_val_dict['loc'] = loc
            op_val_dict['dram'] = dram
            op_val_dict['df'] = out_df
        else:
            op_val_dict['inputs'] = inputs
            op_val_dict['in_df'] = in_df
            op_val_dict['out_df'] = out_df
            op_val_dict['intermed_df'] = intermed_df
            op_val_dict['acc_df'] = acc_df
            op_val_dict['math_fidelity'] = math_fidelity

        if(type == 'matmul'):
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
        if(type == 'queue' or type == 'ram'):
            self.doc['queues'][name] = op_val_dict

    def dump_netlist(self):
        file = yaml.dump(self.doc, sort_keys = True, default_flow_style=True, width=1000)
        print(file)
        with open('res.yaml', 'w') as yaml_file:
            yaml.dump(self.doc, yaml_file, default_flow_style=True)


def main():
    print("Testing TT netlist api")

    simd0  = tt_simd_cluster(2,2,(0,1,2,3))
    num_alloc_blocks = 100000
    alloc0 = tt_malloc(32, num_alloc_blocks, 0)

    netlist0 = tt_netlist()

    block_size = 128
    dim_list0 = (1,2,3,8,8)
    dim_list1 = (1,2,3,4,4)
    dim_list2 = (1,2,3,2,2)

    bla0 = tt_tensor(block_size=block_size, allocator=alloc0, simd_cluster=simd0, torch_tensor=None, shape=tuple(dim_list0))
    bla1 = tt_tensor(block_size=block_size, allocator=alloc0, simd_cluster=simd0, torch_tensor=None, shape=tuple(dim_list1))
    bla2 = tt_tensor(block_size=block_size, allocator=alloc0, simd_cluster=simd0, torch_tensor=None, shape=tuple(dim_list2))

    current_slice = (0,1,2)
    netlist0.add_queues(name='tigrutin', tensors=[bla0,bla1,bla2], current_slice=current_slice)

    netlist0.add_queues(name='fahrutin', tensors=[bla2], current_slice=current_slice)

    netlist0.dump_netlist()

    del(bla0)
    del(bla1)
    del(bla2)

if __name__ == "__main__":
    main()
