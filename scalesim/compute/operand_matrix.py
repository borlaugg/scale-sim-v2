import math
import numpy as np
from tqdm import tqdm

from scalesim.topology_utils import topologies as topoutil
from scale_config import scale_config as cfg
from solver_utils import solver


# This class defines data types for operand matrices
class operand_matrix(object):
    def __init__(self):
        # Objects from outer container classes
        self.config = cfg()
        # self.topoutil = topoutil()
        self.solver = solver()

        # # Layer hyper parameters
        # self.layer_id = 0
        # self.ifmap_rows, self.ifmap_cols = 1, 1
        # self.filter_rows, self.filter_cols = 1, 1
        # self.num_input_channels, self.num_filters = 1, 1
        # self.row_stride, self.col_stride = 1, 1
        # self.batch_size = 1

        # Layer hyper parameters
        self.layer_id = 0
        self.A_rows, self.A_cols = 1, 1
        self.b_rows, self.b_cols = 1, 1
        self.xin_rows, self.xin_cols = 1, 1
        self.num_input_channels, self.num_iterations = 1, 1
        self.row_stride, self.col_stride = 1, 1

        # #  Derived hyper parameters
        # self.ofmap_px_per_filt, self.conv_window_size = 1, 1
        # self.ofmap_rows, self.ofmap_cols = 1, 1

        #  Output hyper parameters
        self.x_rows, self.x_cols = 1, 1

        # # Offsets
        # self.ifmap_offset, self.filter_offset, self.ofmap_offset = 0, 10000000, 20000000
        # self.matrix_offset_arr = [0, 10000000, 20000000]

        # Offsets
        self.A_offset, self.b_offset, self.xin_offset, self.x_offset = 0, 10000000, 20000000, 30000000
        self.matrix_offset_arr = [0, 10000000, 20000000, 30000000]

        # # Address matrices
        # self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.conv_window_size), dtype=int)
        # self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype=int)
        # self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype=int)

        # Address matrices
        self.A_addr_matrix = np.ones((self.A_rows,self.A_cols), dtype='>i4')
        self.b_addr_matrix = np.ones((self.b_rows,self.b_cols), dtype='>i4')
        self.x_addr_matrix = np.ones((self.x_rows,self.x_cols), dtype='>i4')
        self.xin_addr_matrix = np.ones((self.xin_rows,self.xin_cols), dtype='>i4')

        # Flags
        self.params_set_flag = False
        self.matrices_ready_flag = False

    #
    def set_params(self,
                   config_obj,
                   topoutil_obj,
                   layer_id=0,
                   ):

        self.config = config_obj
        self.topoutil = topoutil_obj
        self.layer_id = layer_id

        # TODO: Marked for cleanup
        #my_name = 'operand_matrix.set_params(): '
        #err_prefix = 'Error: ' + my_name
        #
        #if (not len(layer_hyper_param_arr) == 7 and not len(layer_hyper_param_arr) == 8
        #        and not len(layer_hyper_param_arr) == 9) or (not len(layer_calc_hyper_param_arr) == 4) \
        #        or (not len(self.matrix_offset_arr) == 3):
        #    message = err_prefix + 'Invalid arguments. Exiting.'
        #    print(message)
        #    return -1

        self.ifmap_rows, self.ifmap_cols = self.topoutil.get_layer_ifmap_dims(self.layer_id)
        self.filter_rows, self.filter_cols = self.topoutil.get_layer_filter_dims(self.layer_id)
        self.num_input_channels = self.topoutil.get_layer_num_channels(self.layer_id)
        self.num_filters = self.topoutil.get_layer_num_filters(self.layer_id)
        self.row_stride, self.col_stride = self.topoutil.get_layer_strides(self.layer_id)
        # TODO: Marked for cleanup
        #self.row_stride = layer_hyper_param_arr[6]
        #if len(layer_hyper_param_arr) == 8:
        #    self.col_stride = layer_hyper_param_arr[7]

        # TODO: Anand
        # TODO: Next release
        # TODO: Add an option for batching
        self.batch_size = 1

        # TODO: Marked for cleanup
        #if len(layer_hyper_param_arr) == 9:
        #    self.batch_size = layer_hyper_param_arr[8]

        # Assign the calculated hyper parameters
        self.ofmap_rows, self.ofmap_cols = self.topoutil.get_layer_ofmap_dims(self.layer_id)
        self.ofmap_rows = int(self.ofmap_rows)
        self.ofmap_cols = int(self.ofmap_cols)
        self.ofmap_px_per_filt = int(self.ofmap_rows * self.ofmap_cols)
        self.conv_window_size = int(self.topoutil.get_layer_window_size(self.layer_id))

        # Assign the offsets
        self.ifmap_offset, self.filter_offset, self.ofmap_offset \
            = self.config.get_offsets()

        # Address matrices: This is needed to take into account the updated dimensions
        self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt * self.batch_size, self.conv_window_size), dtype='>i4')
        self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype='>i4')
        self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype='>i4')
        self.params_set_flag = True

        # TODO: This should be called from top level
        # TODO: Implement get() function for getting the matrix
        # TODO: Marked for cleanup
        # Return 0 if operand matrix generation is successful
        #self.create_operand_matrices()
        #if self.matrices_ready_flag:
        #    return True, self.ifmap_addr_matrix, self.filter_addr_matrix, self.ofmap_addr_matrix
        #else:
        #    message = err_prefix + 'Address Matrices not created. Exiting!'
        #    print(message)
        #    return False, None, None, None

    def set_solverParams(self,
                   config_obj,
                   solver_obj,
                   layer_id=0,
                   ):

        self.config = config_obj
        self.solver = solver_obj
        self.layer_id = layer_id

        self.A_rows, self.A_cols = self.config.get_A_dims(self.layer_id)
        self.b_rows, self.b_cols = self.config.get_b_dims(self.layer_id)
        self.xin_rows, self.xin_cols = self.config.get_xin_dims(self.layer_id)
        

        # Assign the calculated hyper parameters
        self.x_rows, self.x_cols = self.config.get_x_dims(self.layer_id)

        # Assign the offsets
        self.A_offset, self.b_offset, self.x_offset, self.xin_offset \
            = self.config.get_offsets()

        # Address matrices: This is needed to take into account the updated dimensions
        self.A_addr_matrix = np.ones((self.A_rows,self.A_cols), dtype='>i4')
        self.b_addr_matrix = np.ones((self.b_rows,self.b_cols), dtype='>i4')
        self.x_addr_matrix = np.ones((self.x_rows,self.x_cols), dtype='>i4')
        self.xin_addr_matrix = np.ones((self.xin_rows,self.xin_cols), dtype='>i4')
        self.params_set_flag = True

    # top level function to create the operand matrices
    def create_operand_matrices(self):
        my_name = 'operand_matrix.create_operand_matrices(): '
        err_prefix = 'Error: ' + my_name

        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        # retcode_1 = self.create_ifmap_matrix()
        # retcode_2 = self.create_filter_matrix()
        # retcode_3 = self.create_ofmap_matrix()

        retcode_1 = self.create_A_matrix()
        retcode_2 = self.create_b_matrix()
        retcode_3 = self.create_xin_matrix()
        retcode_4 = self.create_x_matrix()

        retcode = retcode_1 + retcode_2 + retcode_3 + retcode_4
        if retcode == 0:
            self.matrices_ready_flag = True

        return retcode

    # creates the A operand
    def create_A_matrix(self):
        my_name = 'operand_matrix.create_A_matrix(): '
        err_prefix = 'Error: ' + my_name

        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        row_indices = np.arange(self.A_rows)
        col_indices = np.arange(self.A_cols)
        # Create 2D index arrays using meshgrid
        i, j = np.meshgrid(row_indices, col_indices, indexing='ij')

        # Call calc_ifmap_elem_addr_numpy with 2D index arrays
        self.A_addr_matrix = self.calc_A_elem_addr(i, j)
        return 0

    # logic to translate ifmap into matrix fed into systolic array MACs
    def calc_A_elem_addr(self, i, j):
        offset = self.A_offset
        A_rows = self.A_rows
        A_cols = self.A_cols
        A_px_addr = i * A_rows + j + offset

        return A_px_addr

    # creates the x operand
    def create_x_matrix(self):
        my_name = 'operand_matrix.create_x_matrix(): '
        err_prefix = 'Error: ' + my_name
        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        row_indices = np.arange(self.x_rows)
        col_indices = np.arange(1)
        self.x_addr_matrix = self.calc_x_elem_addr(row_indices, col_indices).reshape(self.x_rows, 1)

        return 0

    # logic to translate ofmap into matrix resulting systolic array MACs
    def calc_x_elem_addr(self, i, j):
        offset = self.x_offset
        internal_address = i + j
        x_px_addr = internal_address + offset
        return x_px_addr

    # creates the b operand
    def create_b_matrix(self):
        my_name = 'operand_matrix.create_b_matrix(): '
        err_prefix = 'Error: ' + my_name
        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        row_indices = np.arange(self.b_rows)
        col_indices = np.arange(1)
        self.b_addr_matrix = self.calc_b_elem_addr(row_indices, col_indices).reshape(self.b_rows, 1)

        return 0

    # logic to translate filter into matrix fed into systolic array MACs
    def calc_b_elem_addr(self, i, j):
        offset = self.b_offset
        internal_address = j + i
        b_px_addr = internal_address + offset
        return b_px_addr

    # creates the xin operand
    def create_xin_matrix(self):
        my_name = 'operand_matrix.create_xin_matrix(): '
        err_prefix = 'Error: ' + my_name
        if not self.params_set_flag:
            message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
            print(message)
            return -1

        row_indices = np.arange(self.xin_rows)
        col_indices = np.arange(1)
        self.xin_addr_matrix = self.calc_xin_elem_addr(row_indices, col_indices).reshape(self.xin_rows, 1)

        return 0

    # logic to translate filter into matrix fed into systolic array MACs
    def calc_xin_elem_addr(self, i, j):
        offset = self.xin_offset
        internal_address = j + i
        xin_px_addr = internal_address + offset
        return xin_px_addr

    # function to get a part or the full A operand
    def get_A_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                              num_cols=-1):
        if num_rows == -1:
            num_rows = self.A_rows
        if num_cols == -1:
            num_cols = self.A_cols
        my_name = 'operand_matrix.get_A_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return np.zeros((1, 1))
        if (start_row + num_rows) > self.A_rows or (start_col + num_cols) > self.A_cols:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return np.zeros((1, 1))

        # Anand: ISSUE #3. Patch
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        #ret_mat = self.ifmap_addr_matrix[start_row: end_row][start_col: end_col]
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        ret_mat = self.A_addr_matrix[start_row: end_row, start_col: end_col]
        return ret_mat

    def get_A_matrix(self):
        return self.get_A_matrix_part()

    # function to get a part or the full b operand
    def get_b_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                               num_cols=-1):

        if num_rows == -1:
            num_rows = self.b_rows
        if num_cols == -1:
            num_cols = self.b_cols
        my_name = 'operand_matrix.get_b_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return np.zeros((1, 1))
        if (start_row + num_rows) > self.b_rows:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return np.zeros((1, 1))

        # Anand: ISSUE #3. FIX
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        end_row = start_row + num_rows
        end_col = start_col + num_cols

        # Anand: ISSUE #3. FIX
        #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
        ret_mat = self.b_addr_matrix[start_row: end_row]
        return ret_mat

    def get_b_matrix(self):
        return self.get_b_matrix_part()

    # function to get a part or the full x operand
    def get_x_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                               num_cols=-1):

        # Since we cannot pass self as an argument in the member functions
        # This is an alternate way of making the matrix dimensions as defaults
        if num_rows == -1:
            num_rows = self.x_rows
        if num_cols == -1:
            num_cols = self.x_cols
        my_name = 'operand_matrix.get_x_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return np.zeros((1, 1))
        if (start_row + num_rows) > self.x_rows:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return np.zeros((1, 1))

        # Anand: ISSUE #3. Patch
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        # Anand: ISSUE #7. Patch
        #ret_mat = self.filter_addr_matrix[start_row: end_row, start_col: end_col]
        ret_mat = self.x_addr_matrix[start_row: end_row]

        return ret_mat

    def get_x_matrix(self):
        return self.get_x_matrix_part()

    # function to get a part or the full xin operand
    def get_xin_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
                               num_cols=-1):

        if num_rows == -1:
            num_rows = self.xin_rows
        if num_cols == -1:
            num_cols = self.xin_cols
        my_name = 'operand_matrix.get_xin_matrix_part(): '
        err_prefix = 'Error: ' + my_name
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
                print(message)
                return np.zeros((1, 1))
        if (start_row + num_rows) > self.xin_rows:
            message = err_prefix + ": Illegal arguments. Exiting!"
            print(message)
            return np.zeros((1, 1))

        # Anand: ISSUE #3. FIX
        #end_row = start_row + num_rows + 1
        #end_col = start_col + num_cols + 1
        end_row = start_row + num_rows
        end_col = start_col + num_cols

        # Anand: ISSUE #3. FIX
        #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
        ret_mat = self.xin_addr_matrix[start_row: end_row]
        return ret_mat

    def get_xin_matrix(self):
        return self.get_xin_matrix_part()

    def get_all_operand_matrix(self):
        if not self.matrices_ready_flag:
            me = 'operand_matrix.' + 'get_all_operand_matrix()'
            message = 'ERROR:' + me + ': Matrices not ready or matrix gen failed'
            print(message)
            return

        return self.A_addr_matrix, \
               self.b_addr_matrix, \
               self.x_addr_matrix, \
               self.xin_addr_matrix


if __name__ == '__main__':
    opmat = operand_matrix()
    # tutil = topoutil()
    # lid = 3
    # topology_file = "../../topologies/mlperf/test.csv"
    # tutil.load_arrays(topofile=topology_file)
    # for i in range(tutil.get_num_layers()):
    #     layer_param_arr = tutil.get_layer_params(layer_id=i)
    #     ofmap_dims = tutil.get_layer_ofmap_dims(layer_id=i)
    #     ofmap_px_filt = tutil.get_layer_num_ofmap_px(layer_id=i) / tutil.get_layer_num_filters(layer_id=i)
    #     conv_window_size = tutil.get_layer_window_size(layer_id=i)
    #     layer_calc_hyper_param_arr = [ofmap_dims[0], ofmap_dims[1], ofmap_px_filt, conv_window_size]
    #     config_arr = [512, 512, 256, 8, 8]
        #[matrix_set, ifmap_addr_matrix, filter_addr_matrix, ofmap_addr_matrix] \
        #    = opmat.set_params(layer_hyper_param_arr=layer_param_arr[1:],
        #                       layer_calc_hyper_param_arr=layer_calc_hyper_param_arr,
        #                       offset_list=[0, 1000000, 2000000])
