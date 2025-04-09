import math
import time
import numpy as np
from tqdm import tqdm
from scale_config import scale_config as cfg


class systolic_compute_solver:
    def __init__(self):
        # Params set by user
        self.config = cfg()

        self.A_op_mat = np.zeros((1,1))
        self.b_op_mat = np.zeros((1, 1))
        self.x_op_mat = np.zeros((1, 1))
        self.xin_op_mat = np.zeros((1, 1))

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.arr_row = 0
        self.arr_col = 0

        # Generated matrices
        self.A_prefetch_matrix = np.zeros((1,1))
        self.b_prefetch_matrix = np.zeros((1,1))

        self.A_demand_matrix = np.zeros((1,1))
        self.x_demand_matrix = np.zeros((1,1))
        self.xin_demand_matrix = np.zeros((1,1))
        self.b_demand_matrix = np.zeros((1,1))

        # Generated metrics
        self.A_reads = 0
        self.b_reads = 0
        self.xin_reads = 0
        self.x_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   A_op_mat = np.zeros((1,1)),
                   b_op_mat = np.zeros((1,1)),
                   x_op_mat = np.zeros((1,1)),
                   xin_op_mat = np.zeros((1,1))
                ):

        self.config = config_obj
        self.A_op_mat = A_op_mat
        self.b_op_mat = b_op_mat
        self.x_op_mat = x_op_mat
        self.xin_op_mat = xin_op_mat

        A_col = self.A_op_mat.shape[1]
        b_row= self.b_op_mat.shape[0]

        assert A_col == b_row, "Dimension mismatch between operands"

        self.Sr = self.A_op_mat.shape[0]
        self.Sc = self.A_op_mat.shape[1]
        self.T = self.A_op_mat.shape[0]

        self.arr_row, self.arr_col = self.config.get_array_dims()

        self.col_fold = math.ceil(self.Sc / self.arr_col)
        self.row_fold = math.ceil(self.Sr / self.arr_row)

        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_A_prefetch_mat()
        self.create_b_prefetch_mat()
        self.create_xin_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    def create_A_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'
        for fr in range(self.row_fold):
            for fc in range(self.col_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)
                this_fold_prefetch = self.A_op_mat[row_start_id: row_end_idx, col_start_id: col_end_idx]

                # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
                if col_delta > 0:
                    null_req_mat = np.zeros((this_fold_prefetch.shape[0], col_delta))
                    this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta, this_fold_prefetch.shape[1]))
                    this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=0)

                if fr == 0 and fc == 0:
                    self.A_prefetch_matrix = this_fold_prefetch
                else:
                    self.A_prefetch_matrix = np.concatenate((self.A_prefetch_matrix, this_fold_prefetch), axis=0)

        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand
        #print('DEBUG: create_ifmap_prefetch_mat()')
        #start_time = time.time()

        M, N = self.A_prefetch_matrix.shape
        num_elems = M * N 
        num_diags = M + N
        prefetches = np.zeros((1,num_elems))

        pbar = tqdm(total=M*N, disable=True)
        #print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

        for col_id in range(N):
            for row_id in range(M):
                prefetches[0, row_id * N + col_id] = self.A_prefetch_matrix[row_id][col_id]
                pbar.update(1)

        pbar.close()
        self.A_prefetch_matrix = prefetches

#
    def create_xin_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        for fc in range(self.row_fold):
            for fr in range(self.col_fold):
                row_start_id = fr * self.arr_col
                row_end_idx = min(row_start_id + self.arr_col, self.Sr)
                delta = self.arr_col - (row_end_idx - row_start_id)

                this_fold_prefetch = self.xin_op_mat[row_start_id: row_end_idx]

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((delta,1)) * -1
                    this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=0)

                if fr == 0 and fc == 0:
                    self.xin_prefetch_matrix = this_fold_prefetch
                else:
                    self.xin_prefetch_matrix = np.concatenate((self.xin_prefetch_matrix, this_fold_prefetch), axis=0)


        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand
        #print('DEBUG: create_filter_prefetch_mat()')
        #start_time = time.time()

        # M = self.xin_prefetch_matrix.shape[0]
        # N = 1
        # num_elems = M * N
        # num_diags = M + N
        # prefetches = np.zeros((1, num_elems))
        # idx = 0

        # pbar = tqdm(total=M * N, disable=True)
        # # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

        # for diag_id in range(num_diags):
        #     max_row_id = min(diag_id, M - 1)
        #     min_row_id = max(0, diag_id - N + 1)
        #     valid_rows = max_row_id - min_row_id + 1

        #     for offset in range(valid_rows):
        #         row_id = max_row_id - offset
        #         col_id = diag_id - row_id

        #         elem = self.xin_prefetch_matrix[row_id][col_id]
        #         prefetches[0, idx] = elem
        #         idx += 1
        #         pbar.update(1)

        # pbar.close()
        # self.xin_prefetch_matrix = prefetches

        #t = time.time() - start_time
        #print('DEBUG: create_filter_prefetch_mat =' + str(t))

    #
    def create_b_prefetch_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (row_end_idx - row_start_id)

                this_fold_prefetch = self.b_op_mat[row_start_id: row_end_idx]

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((delta,1)) * -1
                    this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=0)

                if fr == 0 and fc == 0:
                    self.b_prefetch_matrix = this_fold_prefetch
                else:
                    self.b_prefetch_matrix = np.concatenate((self.b_prefetch_matrix, this_fold_prefetch), axis=1)


        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand
        #print('DEBUG: create_filter_prefetch_mat()')
        #start_time = time.time()

        # M = self.b_prefetch_matrix.shape[0]
        # N = 1
        # num_elems = M * N
        # num_diags = M + N
        # prefetches = np.zeros((1, num_elems))
        # idx = 0

        # pbar = tqdm(total=M * N, disable=True)
        # # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

        # for diag_id in range(num_diags):
        #     max_row_id = min(diag_id, M - 1)
        #     min_row_id = max(0, diag_id - N + 1)
        #     valid_rows = max_row_id - min_row_id + 1

        #     for offset in range(valid_rows):
        #         row_id = max_row_id - offset
        #         col_id = diag_id - row_id

        #         elem = self.b_prefetch_matrix[row_id][col_id]
        #         prefetches[0, idx] = elem
        #         idx += 1
        #         pbar.update(1)

        # pbar.close()
        # self.b_prefetch_matrix = prefetches

        #t = time.time() - start_time
        #print('DEBUG: create_filter_prefetch_mat =' + str(t))

    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_A_demand_mat()
        self.create_b_demand_mat()
        self.create_x_demand_mat()
        self.create_xin_demand_mat()

        self.demand_mat_ready_flag = True

    #
    def create_xin_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        # Debug messages
        #print('DEBUG: create_filter_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.row_fold):
            for fr in range(self.col_fold):
                row_start_id = fr * self.arr_col
                row_end_idx = min(row_start_id + self.arr_col, self.Sr)
                delta = self.arr_col - (row_end_idx - row_start_id)

                this_fold_demand = self.xin_op_mat[row_start_id: row_end_idx]
                self.xin_reads += this_fold_demand.shape[0] * 1

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((delta,1)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)
                
                this_fold_demand = np.reshape(this_fold_demand,(1,this_fold_demand.shape[0]))

                if fr == 0 and fc == 0:
                    self.xin_demand_matrix = this_fold_demand
                else:
                    self.xin_demand_matrix = np.concatenate((self.xin_demand_matrix, this_fold_demand), axis=0)

                pbar.update(1)

        pbar.close()
        # TODO: cleanup
        # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
        #self.ofmap_demand_matrix = skew_matrix(self.ofmap_demand_matrix)

    def create_A_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        # Anand: Concatenation issue fix
        inter_fold_gap_suffix = self.arr_row - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        # DEBUG section
        #print('DEBUG: create_ifmap_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fr in range(self.row_fold):
            for fc in range(self.col_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.A_op_mat[row_start_id: row_end_idx, col_start_id: col_end_idx]
                self.A_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta))*-1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta, this_fold_demand.shape[1]))*-1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                this_fold_demand = np.reshape(this_fold_demand,(1,this_fold_demand.shape[0] * this_fold_demand.shape[1]))

                if fr == 0 and fc == 0:
                    self.A_demand_matrix = this_fold_demand
                else:
                    self.A_demand_matrix = np.concatenate((self.A_demand_matrix, this_fold_demand), axis=0)

                pbar.update(1)

        pbar.close()
        # TODO: cleanup
        # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
        #self.ifmap_demand_matrix = skew_matrix(self.ifmap_demand_matrix)

    #
    def create_b_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        # Debug messages
        #print('DEBUG: create_filter_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fr in range(self.row_fold):
            for fc in range(self.col_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (row_end_idx - row_start_id)

                if(fc == 0):
                    this_fold_demand = self.b_op_mat[row_start_id: row_end_idx]
                    self.b_reads += this_fold_demand.shape[0] * 1
                else:
                    this_fold_demand = np.ones(self.b_op_mat[row_start_id: row_end_idx].shape) * -1
                    self.b_reads += 0
                
                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((delta,1)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # In this computation scheme we are allowing the generated outputs to drain out before
                # starting the next fold
                # This portion accounts for that extra time by adding null requests
                # this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)
                this_fold_demand = np.reshape(this_fold_demand,(1,this_fold_demand.shape[0]))

                if fr == 0 and fc == 0:
                    self.b_demand_matrix = this_fold_demand
                else:
                    self.b_demand_matrix = np.concatenate((self.b_demand_matrix, this_fold_demand), axis=0)

                pbar.update(1)

        pbar.close()
        # TODO: Cleanup
        # Add skew to the Filter demand matrix to reflect systolic pipeline fill
        #self.filter_demand_matrix = skew_matrix(self.filter_demand_matrix)

    #
    def create_x_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        # Debug messages
        #print('DEBUG: create_filter_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fr in range(self.row_fold):
            for fc in range(self.col_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (row_end_idx - row_start_id)

                if(fc == 0):
                    this_fold_demand = self.x_op_mat[row_start_id: row_end_idx]
                    self.x_writes += this_fold_demand.shape[0] * 1
                else:
                    this_fold_demand = np.ones(self.x_op_mat[row_start_id: row_end_idx].shape) * -1
                    self.x_writes += 0

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((delta,1)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # Calculate the mapping efficiency
                col_used = 1
                row_used = min(self.arr_row, row_end_idx - row_start_id)
                mac_used = row_used * col_used 
                mapping_eff_this_fold = mac_used / (self.arr_row)

                cycles_this_fold = self.arr_row
                compute_cycles_this_fold = mac_used * cycles_this_fold
                compute_util_this_fold = compute_cycles_this_fold / (self.arr_row * cycles_this_fold)

                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)

                this_fold_demand = np.reshape(this_fold_demand,(1,this_fold_demand.shape[0]))

                if fr == 0 and fc == 0:
                    self.x_demand_matrix = this_fold_demand
                else:
                    self.x_demand_matrix = np.concatenate((self.x_demand_matrix, this_fold_demand), axis=0)
        
                pbar.update(1)

        pbar.close()
        # TODO: cleanup
        # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
        #self.ofmap_demand_matrix = skew_matrix(self.ofmap_demand_matrix)

    #
    def get_A_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.A_prefetch_matrix

    #
    def get_b_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.b_prefetch_matrix

    #
    def get_xin_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.xin_prefetch_matrix

    #
    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.A_prefetch_matrix, self.b_prefetch_matrix, self.xin_prefetch_matrix

    #
    def get_A_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.A_demand_matrix

    #
    def get_b_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.b_demand_matrix

    #
    def get_x_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.x_demand_matrix

    #
    def get_xin_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.xin_demand_matrix

    #
    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.A_demand_matrix, self.b_demand_matrix, self.x_demand_matrix, self.xin_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    #
    def get_A_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.A_reads

    #
    def get_b_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.b_reads

    #
    def get_x_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.x_writes

    #
    def get_xin_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.xin_reads

#
def skew_matrix(input_matrix_np):
    rows, cols = input_matrix_np.shape

    out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)

    for c in range(cols):
        out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]

    return out_matrix_np
