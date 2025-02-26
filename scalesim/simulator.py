import os

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from solver_utils import solver
from single_layer_sim import single_layer_sim as layer_sim


class simulator:
    def __init__(self):
        self.conf = cfg()
        self.topo = topo()
        self.solver = solver()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        # self.num_layers = 0
        self.num_iteration = 0  # Risha: Upper limit on number of iteration

        self.single_layer_sim_object_list = []

        self.params_set_flag = False
        # self.all_layer_run_done = False
        self.exit_condition = False # Risha: Flag to check if exit condition is acheived

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True
                   ):

        self.conf = config_obj
        self.topo = topo_obj

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True

    # Risha: Setting parameters for the solver
    def set_solverParams(self,
                   config_obj=cfg(),
                   solver_obj=solver(),
                   verbosity=True,
                   save_trace=True
                   ):

        self.conf = config_obj

        self.verbose = verbosity
        self.save_trace = save_trace

        # Calculate inferrable parameters here
        self.num_layers = self.solver.get_num_iter()

        self.params_set_flag = True

    #
    def run(self):
        assert self.params_set_flag, 'Simulator parameters are not set'

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            # this_layer_sim.set_params(layer_id=i,
            #                      config_obj=self.conf,
            #                      topology_obj=self.topo,
            #                      verbose=self.verbose)
            this_layer_sim.set_solverParams(layer_id=i,
                                 config_obj=self.conf,
                                 solver_obj=self.solver,
                                 verbose=self.verbose)

            self.single_layer_sim_object_list.append(this_layer_sim)

        # if not os.path.isdir(self.top_path):
        #     os.mkdir(self.top_path)

        # report_path = self.top_path + '/' + self.conf.get_run_name()
        report_path = self.conf.get_run_name()

        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list:

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
                print('\nRunning iteration ' + str(layer_id))

            single_layer_obj.run()

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                comp_cycles = comp_items[0]
                stall_cycles = comp_items[1]
                util = comp_items[2]
                mapping_eff = comp_items[3]
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                avg_A_bw = avg_bw_items[4]
                avg_b_bw = avg_bw_items[5]
                avg_x_bw = avg_bw_items[6]
                avg_xin_bw = avg_bw_items[7]
                print('Average A DRAM BW: ' + "{:.3f}".format(avg_A_bw) + ' words/cycle')
                print('Average b DRAM BW: ' + "{:.3f}".format(avg_b_bw) + ' words/cycle')
                print('Average x DRAM BW: ' + "{:.3f}".format(avg_x_bw) + ' words/cycle')
                print('Average xin DRAM BW: ' + "{:.3f}".format(avg_x_bw) + ' words/cycle')

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path)
                if self.verbose:
                    print('Done!')

            if(single_layer_obj.check_convergence()):
                print("Converged after", layer_id, "iterations")
                break

        self.all_layer_run_done = True

        self.generate_reports()

    #
    def generate_reports(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = 'LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n'
        compute_report.write(header)

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        header = 'LayerID, Avg A SRAM BW, Avg b SRAM BW, Avg x SRAM BW, Avg xin SRAM BW, '
        header += 'Avg A DRAM BW, Avg b DRAM BW, Avg x DRAM BW,Avg xin DRAM BW,\n'
        bandwidth_report.write(header)

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM A Start Cycle, SRAM A Stop Cycle, SRAM A Reads, '
        header += 'SRAM b Start Cycle, SRAM b Stop Cycle, SRAM b Reads, '
        header += 'SRAM x Start Cycle, SRAM x Stop Cycle, SRAM x Writes, '
        header += 'SRAM xin Start Cycle, SRAM xin Stop Cycle, SRAM xin Reads, '
        header += 'DRAM A Start Cycle, DRAM A Stop Cycle, DRAM A Reads, '
        header += 'DRAM b Start Cycle, DRAM b Stop Cycle, DRAM b Reads, '
        header += 'DRAM x Start Cycle, DRAM x Stop Cycle, DRAM x Writes, '
        header += 'DRAM xin Start Cycle, DRAM x Stop Cycle, DRAM xin Reads,\n'
        detail_report.write(header)

        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            compute_report_items_this_layer = single_layer_obj.get_compute_report_items()
            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ',\n'
            compute_report.write(log)

            bandwidth_report_items_this_layer = single_layer_obj.get_bandwidth_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            detail_report_items_this_layer = single_layer_obj.get_detail_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()

    #
    def get_total_cycles(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            cycles_this_layer = int(layer_obj.get_compute_report_items[0])
            total_cycles += cycles_this_layer

        return total_cycles



