import time
import math
import numpy as np
from tqdm import tqdm

from scalesim.memory.read_buffer import read_buffer as rdbuf
from memory.read_buffer_estimate_bw import ReadBufferEstimateBw as rdbuf_est
from scalesim.memory.read_port import read_port as rdport
from scalesim.memory.write_buffer import write_buffer as wrbuf
from scalesim.memory.write_port import write_port as wrport


class double_buffered_scratchpad:
    def __init__(self):
        self.A_buf = rdbuf()
        self.b_buf = rdbuf()
        self.x_buf = wrbuf()
        self.xin_buf = rdbuf()

        self.A_port = rdport()
        self.b_port = rdport()
        self.x_port = wrport()
        self.xin_port = rdport()

        self.verbose = True

        self.A_trace_matrix = np.zeros((1,1), dtype=int)
        self.b_trace_matrix = np.zeros((1,1), dtype=int)
        self.x_trace_matrix = np.zeros((1,1), dtype=int)
        self.xin_trace_matrix = np.zeros((1,1), dtype=int)

        # Metrics to gather for generating run reports
        self.total_cycles = 0
        self.compute_cycles = 0
        self.stall_cycles = 0

        self.avg_A_dram_bw = 0
        self.avg_b_dram_bw = 0
        self.avg_x_dram_bw = 0
        self.avg_xin_dram_bw = 0

        self.A_sram_start_cycle = 0
        self.A_sram_stop_cycle = 0
        self.b_sram_start_cycle = 0
        self.b_sram_stop_cycle = 0
        self.x_sram_start_cycle = 0
        self.x_sram_stop_cycle = 0
        self.xin_sram_start_cycle = 0
        self.xin_sram_stop_cycle = 0

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

        self.estimate_bandwidth_mode = False,
        self.traces_valid = False
        self.params_valid_flag = True

    #
    def set_params(self,
                   verbose=True,
                   estimate_bandwidth_mode=False,
                   word_size=1,
                   A_buf_size_bytes=2, b_buf_size_bytes=2, x_buf_size_bytes=2, xin_buf_size_bytes=2,
                   rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
                   A_backing_buf_bw=1, b_backing_buf_bw=1, x_backing_buf_bw=1, xin_backing_buf_bw=1):

        self.estimate_bandwidth_mode = estimate_bandwidth_mode

        if self.estimate_bandwidth_mode:
            self.A_buf = rdbuf_est()
            self.b_buf = rdbuf_est()
            self.xin_buf = rdbuf_est()

            self.A_buf.set_params(backing_buf_obj=self.A_port,
                                      total_size_bytes=A_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_default_bw=A_backing_buf_bw)

            self.b_buf.set_params(backing_buf_obj=self.b_port,
                                       total_size_bytes=b_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_default_bw=b_backing_buf_bw)

            self.xin_buf.set_params(backing_buf_obj=self.xin_port,
                                       total_size_bytes=xin_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_default_bw=xin_backing_buf_bw)
        else:
            self.A_buf = rdbuf()
            self.b_buf = rdbuf()
            self.xin_buf = rdbuf()

            self.A_buf.set_params(backing_buf_obj=self.A_port,
                                      total_size_bytes=A_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_bw=A_backing_buf_bw)

            self.b_buf.set_params(backing_buf_obj=self.b_port,
                                       total_size_bytes=b_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_bw=b_backing_buf_bw)

            self.xin_buf.set_params(backing_buf_obj=self.xin_port,
                                       total_size_bytes=xin_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_bw=xin_backing_buf_bw)

        self.x_buf.set_params(backing_buf_obj=self.x_port,
                                  total_size_bytes=x_buf_size_bytes,
                                  word_size=word_size,
                                  active_buf_frac=wr_buf_active_frac,
                                  backing_buf_bw=x_backing_buf_bw)

        self.verbose = verbose

        self.params_valid_flag = True

    #
    def set_read_buf_prefetch_matrices(self,
                                       A_prefetch_mat=np.zeros((1,1)),
                                       b_prefetch_mat=np.zeros((1,1)),
                                       xin_prefetch_mat=np.zeros((1,1))
                                       ):

        self.A_buf.set_fetch_matrix(A_prefetch_mat)
        self.b_buf.set_fetch_matrix(b_prefetch_mat)
        self.xin_buf.set_fetch_matrix(xin_prefetch_mat)

    #
    def reset_buffer_states(self):

        self.A_buf.reset()
        self.b_buf.reset()
        self.x_buf.reset()
        self.xin_buf.reset()

    # The following are just shell methods for users to control each mem individually
    def service_xin_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.x_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_A_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.A_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_b_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.b_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_x_writes(self,
                             incoming_requests_arr_np,  # 2D array with the requests
                             incoming_cycles_arr):

        out_cycles_arr_np = self.x_buf.service_writes(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_memory_requests(self, A_demand_mat, b_demand_mat, x_demand_mat, xin_demand_mat):
        assert self.params_valid_flag, 'Memories not initialized yet'

        x_lines = x_demand_mat.shape[0]
        PEs = x_demand_mat.shape[1]
        step = math.sqrt(x_lines)

        self.total_cycles = 0
        self.stall_cycles = 0

        A_hit_latency = self.A_buf.get_hit_latency()
        b_hit_latency = self.b_buf.get_hit_latency()
        xin_hit_latency = self.xin_buf.get_hit_latency()

        A_serviced_cycles = []
        b_serviced_cycles = []
        x_serviced_cycles = []
        xin_serviced_cycles = []

        pbar_disable = not self.verbose
        for i in tqdm(range(x_lines), disable=pbar_disable):

            cycle_arr = np.zeros((1,1)) + i + self.stall_cycles

            A_demand_line = np.reshape(A_demand_mat[i],(1,PEs*PEs))
            A_cycle_out = self.A_buf.service_reads(incoming_requests_arr_np=A_demand_line,
                                                            incoming_cycles_arr=cycle_arr)
            A_serviced_cycles += [A_cycle_out[0]]
            A_stalls = A_cycle_out[0] - cycle_arr[0] - A_hit_latency

            
            b_demand_line = np.reshape(b_demand_mat[i, :],(1, PEs))
            b_cycle_out = self.b_buf.service_reads(incoming_requests_arr_np=b_demand_line,
                                                        incoming_cycles_arr=cycle_arr)

            b_serviced_cycles += [b_cycle_out[0]]
            b_stalls = b_cycle_out[0] - cycle_arr[0] - b_hit_latency
            

            xin_demand_line = np.reshape(xin_demand_mat[i],(1, PEs))
            xin_cycle_out = self.xin_buf.service_reads(incoming_requests_arr_np=xin_demand_line,
                                                        incoming_cycles_arr=cycle_arr)
            xin_serviced_cycles += [xin_cycle_out[0]]
            xin_stalls = xin_cycle_out[0] - cycle_arr[0] - xin_hit_latency

            
            x_demand_line = np.reshape(x_demand_mat[i, :],(1, PEs))
            x_cycle_out = self.x_buf.service_writes(incoming_requests_arr_np=x_demand_line,
                                                            incoming_cycles_arr_np=cycle_arr)
            x_serviced_cycles += [x_cycle_out[0]]
            x_stalls = x_cycle_out[0] - cycle_arr[0]
            

            self.stall_cycles += int(max(A_stalls, b_stalls, x_stalls, xin_stalls))

        if self.estimate_bandwidth_mode:
            # IDE shows warning as complete_all_prefetches is not implemented in read_buffer class
            # It is harmless since, in estimate bandwidth mode, read_buffer_estimate_bw is instantiated
            self.A_buf.complete_all_prefetches()
            self.b_buf.complete_all_prefetches()
            self.xin_buf.complete_all_prefetches()

        self.x_buf.empty_all_buffers(x_serviced_cycles[-1])

        # Prepare the traces
        A_services_cycles_np = np.asarray(A_serviced_cycles).reshape((len(A_serviced_cycles), 1))
        self.A_trace_matrix = np.concatenate((A_services_cycles_np, A_demand_mat), axis=1)

        b_services_cycles_np = np.asarray(b_serviced_cycles).reshape((len(b_serviced_cycles), 1))
        self.b_trace_matrix = np.concatenate((b_services_cycles_np, b_demand_mat), axis=1)

        xin_services_cycles_np = np.asarray(xin_serviced_cycles).reshape((len(xin_serviced_cycles), 1))
        self.xin_trace_matrix = np.concatenate((xin_services_cycles_np, xin_demand_mat), axis=1)

        x_services_cycles_np = np.asarray(x_serviced_cycles).reshape((len(x_serviced_cycles), 1))
        self.x_trace_matrix = np.concatenate((x_services_cycles_np, x_demand_mat), axis=1)
        
        self.total_cycles = int(x_serviced_cycles[-1][0])

        # END of serving demands from memory
        self.traces_valid = True

    # This is the trace computation logic of this memory system
    # Anand: This is too complex, perform the serve cycle by cycle for the requests
    def service_memory_requests_old(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        # TODO: assert sanity check
        assert self.params_valid_flag, 'Memories not initialized yet'

        # Logic:
        # Stalls can occur in both read and write portions and interfere with each other
        # We mitigate interference by picking a window in which there are no write stall,
        # ie, there is sufficient free space in the write buffer

        ofmap_lines_remaining = ofmap_demand_mat.shape[0]       # The three demand mats have the same shape though
        start_line_idx = 0
        end_line_idx = 0

        first = True
        cycle_offset = 0
        self.total_cycles = 0
        self.stall_cycles = 0

        # Status bar
        pbar_disable = not self.verbose #or True
        pbar = tqdm(total=ofmap_lines_remaining, disable=pbar_disable)

        avg_read_time_series = []

        while ofmap_lines_remaining > 0:
            loop_start_time = time.time()
            ofmap_free_space = self.ofmap_buf.get_free_space()

            # Find the number of lines till the ofmap_free_space is filled up
            count = 0
            while not count > ofmap_free_space:
                this_line = ofmap_demand_mat[end_line_idx]
                for elem in this_line:
                    if not elem == -1:
                        count += 1

                if not count > ofmap_free_space:
                    end_line_idx += 1
                    # Limit check
                    if not end_line_idx < ofmap_demand_mat.shape[0]:
                        end_line_idx = ofmap_demand_mat.shape[0] - 1
                        count = ofmap_free_space + 1
                else:   # Send request with minimal data ie one line of the requests
                    end_line_idx += 1
            # END of line counting

            num_lines = end_line_idx - start_line_idx + 1
            this_req_cycles_arr = [int(x + cycle_offset) for x in range(num_lines)]
            this_req_cycles_arr_np = np.asarray(this_req_cycles_arr).reshape((num_lines,1))

            this_req_ifmap_demands = ifmap_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_filter_demands = filter_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_ofmap_demands = ofmap_demand_mat[start_line_idx:(end_line_idx + 1), :]

            no_stall_cycles = num_lines     # Since the cycles are consecutive at this point

            time_start = time.time()
            ifmap_cycles_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=this_req_ifmap_demands,
                                                            incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of the incurred stalls when launching demands for filter_reads
            # Note: Stalls incurred on reading line i in ifmap reflect the request cycles for line i+1 in filter
            ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
            ifmap_stalls = ifmap_cycles_out - this_req_cycles_arr_np - ifmap_hit_latency    # Vec - vec - scalar
            ifmap_stalls = np.concatenate((np.zeros((1,1)), ifmap_stalls[0:-1]), axis=0)    # Shift by one row
            this_req_cycles_arr_np = this_req_cycles_arr_np + ifmap_stalls

            time_start = time.time()
            filter_cycles_out = self.filter_buf.service_reads(incoming_requests_arr_np=this_req_filter_demands,
                                                              incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of stalls again --> The entire array stops when there is a stall
            filter_hit_latency = self.filter_buf.get_hit_latency()
            filter_stalls = filter_cycles_out - this_req_cycles_arr_np - filter_hit_latency  # Vec - vec - scalar
            filter_stalls = np.concatenate((np.zeros((1, 1)), filter_stalls[0:-1]), axis=0)  # Shift by one row
            this_req_cycles_arr_np = this_req_cycles_arr_np + filter_stalls

            ofmap_cycles_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=this_req_ofmap_demands,
                                                             incoming_cycles_arr_np=this_req_cycles_arr_np)

            # Make the trace matrices
            this_req_ifmap_trace_matrix = np.concatenate((ifmap_cycles_out, this_req_ifmap_demands), axis=1)
            this_req_filter_trace_matrix = np.concatenate((filter_cycles_out, this_req_filter_demands), axis=1)
            this_req_ofmap_trace_matrix = np.concatenate((ofmap_cycles_out, this_req_ofmap_demands), axis=1)

            actual_cycles = ofmap_cycles_out[-1][0] - this_req_cycles_arr_np[0][0] + 1
            num_stalls = actual_cycles - no_stall_cycles

            self.stall_cycles += num_stalls
            self.total_cycles = ofmap_cycles_out[-1][0] + 1         # OFMAP is served the last

            if first:
                first = False
                self.ifmap_trace_matrix = this_req_ifmap_trace_matrix
                self.filter_trace_matrix = this_req_filter_trace_matrix
                self.ofmap_trace_matrix = this_req_ofmap_trace_matrix
            else:
                self.ifmap_trace_matrix = np.concatenate((self.ifmap_trace_matrix, this_req_ifmap_trace_matrix), axis=0)
                self.filter_trace_matrix = np.concatenate((self.filter_trace_matrix, this_req_filter_trace_matrix), axis=0)
                self.ofmap_trace_matrix = np.concatenate((self.ofmap_trace_matrix, this_req_ofmap_trace_matrix), axis=0)

            # Update the local variable for another iteration of the while loop
            cycle_offset = ofmap_cycles_out[-1][0] + 1
            start_line_idx = end_line_idx + 1

            pbar.update(num_lines)
            ofmap_lines_remaining = max(ofmap_demand_mat.shape[0] - (end_line_idx + 1), 0)    # Cutoff at 0
            #print("DEBUG: " + str(end_line_idx))

            if end_line_idx > ofmap_demand_mat.shape[0]:
                print('Trap')

            #if int(ofmap_lines_remaining % 1000) == 0:
            #    print("DEBUG: " + str(ofmap_lines_remaining))

            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            #print('DEBUG: Time taken in one iteration: ' + str(loop_time))

        # At this stage there might still be some data in the active buffer of the OFMAP scratchpad
        # The following drains it and generates the OFMAP
        drain_start_cycle = self.ofmap_trace_matrix[-1][0] + 1
        self.ofmap_buf.empty_all_buffers(drain_start_cycle)

        #avg_read_time = sum(avg_read_time_series) / len(avg_read_time_series)
        #print('DEBUG: Avg time to service reads= ' + str(avg_read_time))

        pbar.close()
        # END of serving demands from memory
        self.traces_valid = True

    #
    def get_total_compute_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.total_cycles

    #
    def get_stall_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.stall_cycles

    #
    def get_A_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.A_trace_matrix.shape[0]):
            if done:
                break
            row = self.A_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.A_sram_start_cycle = self.A_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.A_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.A_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.A_sram_stop_cycle  = self.A_trace_matrix[ridx][0]
                    done = True
                    break

        return self.A_sram_start_cycle, self.A_sram_stop_cycle

    #
    def get_b_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.b_trace_matrix.shape[0]):
            if done:
                break
            row = self.b_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.b_sram_start_cycle = self.b_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.b_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.b_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.b_sram_stop_cycle = self.b_trace_matrix[ridx][0]
                    done = True
                    break

        return self.b_sram_start_cycle, self.b_sram_stop_cycle

    #
    def get_x_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.x_trace_matrix.shape[0]):
            if done:
                break
            row = self.x_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.x_sram_start_cycle = self.x_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.x_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.x_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.x_sram_stop_cycle = self.x_trace_matrix[ridx][0]
                    done = True
                    break

        return self.x_sram_start_cycle, self.x_sram_stop_cycle

    #
    def get_xin_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.x_trace_matrix.shape[0]):
            if done:
                break
            row = self.xin_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.xin_sram_start_cycle = self.xin_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.xin_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.xin_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.xin_sram_stop_cycle = self.xin_trace_matrix[ridx][0]
                    done = True
                    break

        return self.xin_sram_start_cycle, self.xin_sram_stop_cycle

    #
    def get_A_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.A_dram_reads = self.A_buf.get_num_accesses()
        self.A_dram_start_cycle, self.A_dram_stop_cycle \
            = self.A_buf.get_external_access_start_stop_cycles()

        return self.A_dram_start_cycle, self.A_dram_stop_cycle, self.A_dram_reads

    #
    def get_b_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.b_dram_reads = self.b_buf.get_num_accesses()
        self.b_dram_start_cycle, self.b_dram_stop_cycle \
            = self.b_buf.get_external_access_start_stop_cycles()

        return self.b_dram_start_cycle, self.b_dram_stop_cycle, self.b_dram_reads

   #
    def get_xin_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.xin_dram_reads = self.xin_buf.get_num_accesses()
        self.xin_dram_start_cycle, self.xin_dram_stop_cycle \
            = self.xin_buf.get_external_access_start_stop_cycles()

        return self.xin_dram_start_cycle, self.xin_dram_stop_cycle, self.xin_dram_reads

    #
    def get_x_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.x_dram_writes = self.x_buf.get_num_accesses()
        self.x_dram_start_cycle, self.x_dram_stop_cycle \
            = self.x_buf.get_external_access_start_stop_cycles()

        return self.x_dram_start_cycle, self.x_dram_stop_cycle, self.x_dram_writes

    #
    def get_A_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.A_trace_matrix

    #
    def get_b_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.b_trace_matrix

    #
    def get_x_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.x_trace_matrix

    #
    def get_xin_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.x_trace_matrix

    #
    def get_sram_trace_matrices(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.A_trace_matrix, self.b_trace_matrix, self.x_trace_matrix, self.xin_trace_matrix

    #
    def get_A_dram_trace_matrix(self):
        return self.A_buf.get_trace_matrix()

    #
    def get_b_dram_trace_matrix(self):
        return self.b_buf.get_trace_matrix()

   #
    def get_xin_dram_trace_matrix(self):
        return self.xin_buf.get_trace_matrix()

    #
    def get_x_dram_trace_matrix(self):
        return self.x_buf.get_trace_matrix()

    #
    def get_dram_trace_matrices(self):
        dram_A_trace = self.A_buf.get_trace_matrix()
        dram_b_trace = self.b_buf.get_trace_matrix()
        dram_xin_trace = self.xin_buf.get_trace_matrix()
        dram_x_trace = self.x_buf.get_trace_matrix()

        return dram_A_trace, dram_b_trace, dram_x_trace, dram_xin_trace

        #
    def print_A_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.A_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_b_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.b_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_x_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.x_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_xin_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.xin_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_A_dram_trace(self, filename):
        self.A_buf.print_trace(filename)

    #
    def print_b_dram_trace(self, filename):
        self.b_buf.print_trace(filename)

    #
    def print_x_dram_trace(self, filename):
        self.x_buf.print_trace(filename)

    #
    def print_xin_dram_trace(self, filename):
        self.xin_buf.print_trace(filename)





