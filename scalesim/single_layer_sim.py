import os

from scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from solver_utils import solver
from compute.operand_matrix import operand_matrix as opmat
from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.compute.systolic_compute_is import systolic_compute_is
from compute.systolic_compute_solver import systolic_compute_solver
from memory.double_buffered_scratchpad_mem import double_buffered_scratchpad as mem_dbsp


class single_layer_sim:
    def __init__(self):
        self.layer_id = 0
        # self.topo = topo()
        self.solver = solver()
        self.config = cfg()

        self.op_mat_obj = opmat()
        # self.compute_system = systolic_compute_os()
        self.compute_system = systolic_compute_solver()
        self.memory_system = mem_dbsp()

        self.verbose = True

        # Report items : Compute report
        self.total_cycles = 0
        self.stall_cycles = 0
        self.num_compute = 0
        self.num_mac_unit = 0
        self.overall_util = 0
        self.mapping_eff = 0
        self.compute_util = 0

        # Report items : BW report
        self.avg_A_sram_bw = 0
        self.avg_b_sram_bw = 0
        self.avg_x_sram_bw = 0
        self.avg_xin_sram_bw = 0
        self.avg_A_dram_bw = 0
        self.avg_b_dram_bw = 0
        self.avg_x_dram_bw = 0
        self.avg_xin_dram_bw = 0

        # Report items : Detailed Access report
        self.A_sram_start_cycle = 0
        self.A_sram_stop_cycle = 0
        self.A_sram_reads = 0

        self.b_sram_start_cycle = 0
        self.b_sram_stop_cycle = 0
        self.b_sram_reads = 0

        self.x_sram_start_cycle = 0
        self.x_sram_stop_cycle = 0
        self.x_sram_writes = 0

        self.xin_sram_start_cycle = 0
        self.xin_sram_stop_cycle = 0
        self.xin_sram_reads = 0

        self.A_dram_start_cycle = 0
        self.A_dram_stop_cycle = 0
        self.A_dram_reads = 0

        self.b_dram_start_cycle = 0
        self.b_dram_stop_cycle = 0
        self.b_dram_reads = 0

        self.x_dram_start_cycle = 0
        self.x_dram_stop_cycle = 0
        self.x_dram_writes = 0

        self.xin_dram_start_cycle = 0
        self.xin_dram_stop_cycle = 0
        self.xin_dram_reads = 0

        self.params_set_flag = False
        self.memory_system_ready_flag = False
        self.runs_ready = False
        self.report_items_ready = False

        self.converged = False

    def set_params(self,
                   layer_id=0,
                   config_obj=cfg(), topology_obj=topo(),
                   verbose=True):

        self.layer_id = layer_id
        self.config = config_obj
        self.topo = topology_obj

        self.op_mat_obj.set_params(layer_id=self.layer_id,
                                   config_obj=self.config,
                                   topoutil_obj=self.topo,
                                   )

        self.dataflow = self.config.get_dataflow()
        if self.dataflow == 'os':
            self.compute_system = systolic_compute_os()
        elif self.dataflow == 'ws':
            self.compute_system = systolic_compute_ws()
        elif self.dataflow == 'is':
            self.compute_system = systolic_compute_is()

        arr_dims =self.config.get_array_dims()
        self.num_mac_unit = arr_dims[0] * arr_dims[1]
        self.verbose=verbose

        self.params_set_flag = True

    def set_solverParams(self,
                   layer_id=0,
                   config_obj=cfg(), solver_obj=solver(),
                   verbose=True):

        self.layer_id = layer_id
        self.config = config_obj
        self.solver = solver_obj

        self.op_mat_obj.set_solverParams(layer_id=self.layer_id,
                                   config_obj=self.config,
                                   solver_obj=self.solver,
                                   )

        self.compute_system = systolic_compute_solver()

        arr_dims =self.config.get_array_dims()
        self.num_mac_unit = arr_dims[0]
        self.verbose=verbose

        self.params_set_flag = True

    # This communicates that the memory is being managed externally
    # And the class will not interfere with setting it up
    def set_memory_system(self, mem_sys_obj=mem_dbsp()):
        self.memory_system = mem_sys_obj
        self.memory_system_ready_flag = True

    def run(self):
        assert self.params_set_flag, 'Parameters are not set. Run set_params()'

        # 1. Setup and the get the demand from compute system

        # 1.1 Get the operand matrices
        # _, ifmap_op_mat = self.op_mat_obj.get_ifmap_matrix()
        # _, filter_op_mat = self.op_mat_obj.get_filter_matrix()
        # _, ofmap_op_mat = self.op_mat_obj.get_ofmap_matrix()
        A_op_mat = self.op_mat_obj.get_A_matrix()
        b_op_mat = self.op_mat_obj.get_b_matrix()
        x_op_mat = self.op_mat_obj.get_x_matrix()
        xin_op_mat = self.op_mat_obj.get_xin_matrix()

        # self.num_compute = self.topo.get_layer_num_ofmap_px(self.layer_id) \
                        #    * self.topo.get_layer_window_size(self.layer_id)
        #TODO check this for correctness
        n, m = self.config.get_array_dims()
        self.num_compute = self.solver.get_num_iter()*2*(pow(n,2))

        # 1.2 Get the prefetch matrices for both operands
        # self.compute_system.set_params(config_obj=self.config,
        #                                ifmap_op_mat=ifmap_op_mat,
        #                                filter_op_mat=filter_op_mat,
        #                                ofmap_op_mat=ofmap_op_mat)
        
        self.compute_system.set_params(config_obj = self.config,
                                       A_op_mat=A_op_mat,
                                       b_op_mat=b_op_mat,
                                       x_op_mat=x_op_mat,
                                       xin_op_mat=xin_op_mat)

        # 1.3 Get the no compute demand matrices from for 2 operands and the output
        A_prefetch_mat, b_prefetch_mat, xin_prefetch_mat = self.compute_system.get_prefetch_matrices()
        A_demand_mat, b_demand_mat, x_demand_mat, xin_demand_matrix = self.compute_system.get_demand_matrices()
        #print('DEBUG: Compute operations done')
        # 2. Setup the memory system and run the demands through it to find any memory bottleneck and generate traces

        
        # 2.1 Setup the memory system if it was not setup externally
        if not self.memory_system_ready_flag:
            word_size = 1           # bytes, this can be incorporated in the config file
            active_buf_frac = 0.5   # This can be incorporated in the config as well

            A_buf_size_kb, b_buf_size_kb, x_buf_size_kb, xin_buf_size_kb = self.config.get_mem_sizes()
            A_buf_size_bytes = 1024 * A_buf_size_kb
            b_buf_size_bytes = 1024 * b_buf_size_kb
            x_buf_size_bytes = 1024 * x_buf_size_kb
            xin_buf_size_bytes = 1024 * xin_buf_size_kb

            A_backing_bw = 1
            b_backing_bw = 1
            x_backing_bw = 1
            xin_backing_bw = 1
            estimate_bandwidth_mode = False
            if self.config.use_user_dram_bandwidth():
                bws = self.config.get_bandwidths_as_list()
                A_backing_bw = bws[0]
                b_backing_bw = bws[0]
                x_backing_bw = bws[0]
                xin_backing_bw = bws[0]

            else:
                dataflow = self.config.get_dataflow()
                arr_row, arr_col = self.config.get_array_dims()
                estimate_bandwidth_mode = True

                # The number 10 elems per cycle is arbitrary
                A_backing_bw = 10
                b_backing_bw = 10
                x_backing_bw = arr_row
                xin_backing_bw = arr_row

            self.memory_system.set_params(
                    word_size=word_size,
                    A_buf_size_bytes=A_buf_size_bytes,
                    b_buf_size_bytes=b_buf_size_bytes,
                    x_buf_size_bytes=x_buf_size_bytes,
                    xin_buf_size_bytes=xin_buf_size_bytes,
                    rd_buf_active_frac=active_buf_frac, wr_buf_active_frac=active_buf_frac,
                    A_backing_buf_bw=A_backing_bw,
                    b_backing_buf_bw=b_backing_bw,
                    x_backing_buf_bw=x_backing_bw,
                    xin_backing_buf_bw=xin_backing_bw,
                    verbose=self.verbose,
                    estimate_bandwidth_mode=estimate_bandwidth_mode
            )

        # 2.2 Install the prefetch matrices to the read buffers to finish setup
        if self.config.use_user_dram_bandwidth() :
            self.memory_system.set_read_buf_prefetch_matrices(A_prefetch_mat=A_prefetch_mat,
                                                              b_prefetch_mat=b_prefetch_mat,
                                                              xin_prefetch_mat=xin_prefetch_mat)

        # 2.3 Start sending the requests through the memory system until
        # all the OFMAP memory requests have been serviced
        self.memory_system.service_memory_requests(A_demand_mat, b_demand_mat, x_demand_mat, xin_demand_matrix)

        self.runs_ready = True

    # This will write the traces
    def save_traces(self, top_path):
        assert self.params_set_flag, 'Parameters are not set'

        dir_name = top_path + '/layer' + str(self.layer_id)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        A_sram_filename = dir_name +  '/A_SRAM_TRACE.csv'
        b_sram_filename = dir_name + '/b_SRAM_TRACE.csv'
        x_sram_filename = dir_name +  '/x_SRAM_TRACE.csv'
        xin_sram_filename = dir_name +  '/xin_SRAM_TRACE.csv'

        A_dram_filename = dir_name +  '/A_DRAM_TRACE.csv'
        b_dram_filename = dir_name + '/b_DRAM_TRACE.csv'
        x_dram_filename = dir_name +  '/x_DRAM_TRACE.csv'
        xin_dram_filename = dir_name +  '/xin_DRAM_TRACE.csv'

        self.memory_system.print_A_sram_trace(A_sram_filename)
        self.memory_system.print_A_dram_trace(A_dram_filename)
        self.memory_system.print_b_sram_trace(b_sram_filename)
        self.memory_system.print_b_dram_trace(b_dram_filename)
        self.memory_system.print_x_sram_trace(x_sram_filename)
        self.memory_system.print_x_dram_trace(x_dram_filename)
        self.memory_system.print_xin_sram_trace(xin_sram_filename)
        self.memory_system.print_xin_dram_trace(xin_dram_filename)

    #
    def calc_report_data(self):
        assert self.runs_ready, 'Runs are not done yet'

        # Compute report
        self.total_cycles = self.memory_system.get_total_compute_cycles()
        self.stall_cycles = self.memory_system.get_stall_cycles()
        self.overall_util = (self.num_compute * 100) / (self.total_cycles * self.num_mac_unit)
        self.mapping_eff = self.compute_system.get_avg_mapping_efficiency() * 100
        self.compute_util = self.compute_system.get_avg_compute_utilization() * 100

        # BW report
        self.A_sram_reads = self.compute_system.get_A_requests()
        self.b_sram_reads = self.compute_system.get_b_requests()
        self.x_sram_writes = self.compute_system.get_x_requests()
        self.xin_sram_reads = self.compute_system.get_xin_requests()
        self.avg_A_sram_bw = self.A_sram_reads / self.total_cycles
        self.avg_b_sram_bw = self.b_sram_reads / self.total_cycles
        self.avg_x_sram_bw = self.x_sram_writes / self.total_cycles
        self.avg_xin_sram_bw = self.xin_sram_reads / self.total_cycles

        # Detail report
        self.A_sram_start_cycle, self.A_sram_stop_cycle \
            = self.memory_system.get_A_sram_start_stop_cycles()

        self.b_sram_start_cycle, self.b_sram_stop_cycle \
            = self.memory_system.get_b_sram_start_stop_cycles()

        self.x_sram_start_cycle, self.x_sram_stop_cycle \
            = self.memory_system.get_x_sram_start_stop_cycles()

        self.xin_sram_start_cycle, self.xin_sram_stop_cycle \
            = self.memory_system.get_xin_sram_start_stop_cycles()

        self.A_dram_start_cycle, self.A_dram_stop_cycle, self.A_dram_reads \
            = self.memory_system.get_A_dram_details()

        self.b_dram_start_cycle, self.b_dram_stop_cycle, self.b_dram_reads \
            = self.memory_system.get_b_dram_details()

        self.x_dram_start_cycle, self.x_dram_stop_cycle, self.x_dram_writes \
            = self.memory_system.get_x_dram_details()

        self.xin_dram_start_cycle, self.xin_dram_stop_cycle, self.xin_dram_reads \
            = self.memory_system.get_xin_dram_details()

        # BW calc for DRAM access
        self.avg_A_dram_bw = self.A_dram_reads / (self.A_dram_stop_cycle - self.A_dram_start_cycle + 1)
        self.avg_b_dram_bw = self.b_dram_reads / (self.b_dram_stop_cycle - self.b_dram_start_cycle + 1)
        self.avg_x_dram_bw = self.x_dram_writes / (self.x_dram_stop_cycle - self.x_dram_start_cycle + 1)
        self.avg_xin_dram_bw = self.xin_dram_reads / (self.xin_dram_stop_cycle - self.xin_dram_start_cycle + 1)

        self.report_items_ready = True

    #
    def get_layer_id(self):
        assert self.params_set_flag, 'Parameters are not set yet'
        return self.layer_id

    #
    def get_compute_report_items(self):
        if not self.report_items_ready:
            self.calc_report_data()

        items = [self.total_cycles, self.stall_cycles, self.overall_util, self.mapping_eff, self.compute_util]
        return items

    #
    def get_bandwidth_report_items(self):
        if not self.report_items_ready:
            self.calc_report_data()

        items = [self.avg_A_sram_bw, self.avg_b_sram_bw, self.avg_x_sram_bw, self.avg_xin_sram_bw]
        items += [self.avg_A_dram_bw, self.avg_b_dram_bw, self.avg_x_dram_bw, self.avg_xin_dram_bw]

        return items

    #
    def get_detail_report_items(self):
        if not self.report_items_ready:
            self.calc_report_data()

        items = [self.A_sram_start_cycle, self.A_sram_stop_cycle, self.A_sram_reads]
        items += [self.b_sram_start_cycle, self.b_sram_stop_cycle, self.b_sram_reads]
        items += [self.x_sram_start_cycle, self.x_sram_stop_cycle, self.x_sram_writes]
        items += [self.xin_sram_start_cycle, self.xin_sram_stop_cycle, self.xin_sram_reads]
        items += [self.A_dram_start_cycle, self.A_dram_stop_cycle, self.A_dram_reads]
        items += [self.b_dram_start_cycle, self.b_dram_stop_cycle, self.b_dram_reads]
        items += [self.x_dram_start_cycle, self.x_dram_stop_cycle, self.x_dram_writes]
        items += [self.xin_dram_start_cycle, self.xin_dram_stop_cycle, self.xin_dram_reads]

        return items

    #
    def check_convergence(self):
        assert self.runs_ready, 'Iteration is not done yet'
        return self.converged
