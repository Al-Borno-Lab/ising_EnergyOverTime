from modules import *
from stat_utility import *
from utility import *

import argparse
"""
Run a comparison of the model multipliers for the stimulus sets of each mouse to see if there is a significant difference
Comparison not implemented in this script

If not significant difference (expected)
    Generate energy vs time for the full reach period, the pre reach period, and the active reach period
    Generate the heat capacity plots for the same periods with the stim0 model multipliers
    Generate the average spin (magnetization) plots 
    Generate the energy curve vs temperature
    Mark out the critical temperature in the temperature plots
    Mark out the critical energy (max energy) in the energy vs time plots
    Check to see if the peak energy matches up with the critical temp energy and at one point in time that energy is reached
    
If there is significant difference
    ???
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--DATA_LOAD_FILE_DIR",
                        default='./Data',
                        help="Directory for loading data files")

    parser.add_argument("--MODEL_SAVE_FILE_DIR",
                        default='./Output/Model',
                        help="Directory for saving output files")

    parser.add_argument("--STAT_SAVE_FILE_DIR",
                        default='./Output/Stats',
                        help="Directory for saving the statistical analysis of the model and data")

    args = parser.parse_args()

    # Stores the directory of the spike data for each mouse
    mice = os.listdir(args.DATA_LOAD_FILE_DIR)
    mice_models_dir = os.listdir(args.MODEL_SAVE_FILE_DIR)

    for mouse in mice:
        # Cells directory
        # cells_dir = os.listdir(f'{args.DATA_LOAD_FILE_DIR}/{mouse}')
        cells_dir = ['Non_P_Cells', 'All_Cells', 'P_Cells']
        model_cell_dir = ['Non_P_Cells', 'All_Cells', 'P_Cells']

        for cell in cells_dir:
            # bin directory: how the data was binned (bin10 = 10 ms window)
            bin_dir = os.listdir(f'{args.DATA_LOAD_FILE_DIR}/{mouse}/{cell}')

            # directory containing all the raw matlab data for neural and kinematic activity
            session_dir = sorted(os.listdir(f'{args.DATA_LOAD_FILE_DIR}/{mouse}/{cell}/{bin_dir[0]}'), key=sort_files)
            session_model_dir = sorted(
                os.listdir(f'{args.MODEL_SAVE_FILE_DIR}/{mouse}/{cell}'), key=sort_files)

            for i in range(len(session_dir)):
                # directory of where the models are saved
                stim_models_dir = sorted(os.listdir(f'{args.MODEL_SAVE_FILE_DIR}/{mouse}/{cell}/{session_model_dir[i]}'), key=sort_files)

                # load matlab data
                mat_data = loadmat(f'{args.DATA_LOAD_FILE_DIR}/{mouse}/{cell}/{bin_dir[0]}/{session_dir[i]}')

                # parse data into spikes and kinematics
                spike_data, kin_data = load_data(mat_data)

                # session_model_dir[index] will pull a given session, stim_models_dir[0] is stim0 model for that session
                # model_stim_files_dir = os.listdir(
                #     f'{args.MODEL_SAVE_FILE_DIR}/{mouse}/{cell}/{session_model_dir[i]}/{stim_models_dir[0]}')
                model_stim_files_dir = 'mid_peak_reach.npy'
                # load full_reach for the stim0 model multipliers represented by model_stim_files_dir[0]
                model = np.load(
                    f'{args.MODEL_SAVE_FILE_DIR}/{mouse}/{cell}/{session_model_dir[i]}/{stim_models_dir[0]}/{model_stim_files_dir}')

                # convert spikes to spins
                spin_data = []
                for stim in range(len(spike_data)):
                    temp_spins = []
                    for reach in range(len(spike_data[stim])):
                        temp_spins += [2 * preprocessingSpikes(spike_data[stim][reach][451:601, :], 1) - 1]

                    spin_data += [temp_spins]

                bin_cat_0 = np.vstack(spin_data[0])
                bin_cat_1 = np.vstack(spin_data[1])
                bin_cat_2 = np.vstack(spin_data[2])
                """
                I think we would need the pre reach and active reach multipliers in order to calculate the heat capacity
                for those time segments
                """
                # phase graph does the metropolis sampling over the temperature range 0 to 3 in steps of 0.05, returns the
                # energy, spins, and temp range calculated/used
                # the energy here is calculated using the metropolis sampling method
                print(f"Calculating energy and spins for stim 0 of {mouse} session {session_dir[i]}")
                avg_stim0_energy, avg_stim0_spins, temp_range_0 = phase_graph(bin_cat_0[0, :], model, n=bin_cat_0.shape[1])
                print(f"Calculating energy and spins for stim 1 of {mouse} session {session_dir[i]}")
                avg_stim1_energy, avg_stim1_spins, temp_range_1 = phase_graph(bin_cat_1[0, :], model, n=bin_cat_1.shape[1])
                print(f"Calculating energy and spins for stim 2 of {mouse} session {session_dir[i]}")
                avg_stim2_energy, avg_stim2_spins, temp_range_2 = phase_graph(bin_cat_2[0, :], model, n=bin_cat_2.shape[1])

                # position_time calculates the average x position over the total time frame for all 3 stims, returns the list
                x_stim0, x_stim1, x_stim2 = position_time(kin_data)

                # energy_time calculates the average energy for the spin data and model multipliers from the Ising model
                # the energy here is calculated using the actual ising model
                en_stim0, en_stim1, en_stim2 = energy_time(spin_data, model)

                avg_energy_0 = [np.mean(r) for r in avg_stim0_energy]
                avg_energy_1 = [np.mean(r) for r in avg_stim1_energy]
                avg_energy_2 = [np.mean(r) for r in avg_stim2_energy]

                avg_spins_0 = [np.mean(r) / bin_cat_0.shape[1] ** 2 for r in avg_stim0_spins]
                avg_spins_1 = [np.mean(r) / bin_cat_1.shape[1] ** 2 for r in avg_stim1_spins]
                avg_spins_2 = [np.mean(r) / bin_cat_2.shape[1] ** 2 for r in avg_stim2_spins]

                # heat_cap is the plot, tc_idx_0 is the index of the crit temp, tc_0 is the energy value at that index
                heat_cap_stim0, Tc_idx_0, tc_0 = plot_heat_capacity(avg_stim0_energy, temp_range_0)
                heat_cap_stim1, Tc_idx_1, tc_1 = plot_heat_capacity(avg_stim1_energy, temp_range_1)
                heat_cap_stim2, Tc_idx_2, tc_2 = plot_heat_capacity(avg_stim2_energy, temp_range_2)

                heat_cap_stim0.gca().set_title('Heat Capacity for Stim 0')
                heat_cap_stim1.gca().set_title('Heat Capacity for Stim 1')
                heat_cap_stim2.gca().set_title('Heat Capacity for Stim 2')

                mag_spin_stim0 = plot_magnetization(avg_spins_0, temp_range_0)
                mag_spin_stim1 = plot_magnetization(avg_spins_1, temp_range_1)
                mag_spin_stim2 = plot_magnetization(avg_spins_2, temp_range_2)

                mag_spin_stim0.gca().set_title('Spin per Temperature for Stim 0')
                mag_spin_stim1.gca().set_title('Spin per Temperature for Stim 1')
                mag_spin_stim2.gca().set_title('Spin per Temperature for Stim 2')

                energy_temp_stim0, crit_en_0, Tc_en_0 = plot_energy(avg_energy_0, temp_range_0, Tc_idx_0)
                energy_temp_stim1, crit_en_1, Tc_en_1 = plot_energy(avg_energy_1, temp_range_1, Tc_idx_1)
                energy_temp_stim2, crit_en_2, Tc_en_2 = plot_energy(avg_energy_2, temp_range_2, Tc_idx_2)

                energy_temp_stim0.gca().set_title('Energy per Temperature for Stim 0')
                energy_temp_stim1.gca().set_title('Energy per Temperature for Stim 1')
                energy_temp_stim2.gca().set_title('Energy per Temperature for Stim 2')

                f_0 = scipy.interpolate.CubicSpline(temp_range_0, avg_energy_0)
                f_1 = scipy.interpolate.CubicSpline(temp_range_1, avg_energy_1)
                f_2 = scipy.interpolate.CubicSpline(temp_range_2, avg_energy_2)

                # en_stim0 is the model multipliers from stim0 applied to data of stim0
                en_pos_plot_stim0 = combine_energy_position(en_stim0, x_stim0, f_0, tc_0, crit_en_0, Tc_en_0)

                # en_stim1 is the model multipliers from stim0 applied to data of stim1
                en_pos_plot_stim1 = combine_energy_position(en_stim1, x_stim1, f_1, tc_1, crit_en_1, Tc_en_1)

                # en_stim2 is the model multipliers from stim0 applied to data of stim2
                en_pos_plot_stim2 = combine_energy_position(en_stim2, x_stim2, f_2, tc_2, crit_en_2, Tc_en_2)

                en_pos_plot_stim0.get_axes()[0].set_title('Distribution of Energy over time for Stim 0')
                en_pos_plot_stim1.get_axes()[0].set_title('Distribution of Energy over time for Stim 1')
                en_pos_plot_stim2.get_axes()[0].set_title('Distribution of Energy over time for Stim 2')

                save_path_energy = f'{args.STAT_SAVE_FILE_DIR}/Mid_Peak_Reach/Plots/Energy/{mouse}/{bin_dir[0]}/{cell}/{session_model_dir[i]}'
                save_path_en_pos = f'{args.STAT_SAVE_FILE_DIR}/Mid_Peak_Reach/Plots/Energy_Position/{mouse}/{bin_dir[0]}/{cell}/{session_model_dir[i]}'
                save_path_heat_cap = f'{args.STAT_SAVE_FILE_DIR}/Mid_Peak_Reach/Plots/Heat_Capacity/{mouse}/{bin_dir[0]}/{cell}/{session_model_dir[i]}'
                save_path_mag_spins = f'{args.STAT_SAVE_FILE_DIR}/Mid_Peak_Reach/Plots/Mag_Spins/{mouse}/{bin_dir[0]}/{cell}/{session_model_dir[i]}'

                if not os.path.exists(save_path_energy):
                    os.makedirs(save_path_energy)

                if not os.path.exists(save_path_en_pos):
                    os.makedirs(save_path_en_pos)

                if not os.path.exists(save_path_heat_cap):
                    os.makedirs(save_path_heat_cap)

                if not os.path.exists(save_path_mag_spins):
                    os.makedirs(save_path_mag_spins)

                en_pos_plot_stim0.savefig(f'{save_path_en_pos}/energy_pos_stim0.png')
                en_pos_plot_stim1.savefig(f'{save_path_en_pos}/energy_pos_stim1.png')
                en_pos_plot_stim2.savefig(f'{save_path_en_pos}/energy_pos_stim2.png')

                heat_cap_stim0.savefig(f'{save_path_heat_cap}/heat_cap_stim0.png')
                heat_cap_stim1.savefig(f'{save_path_heat_cap}/heat_cap_stim1.png')
                heat_cap_stim2.savefig(f'{save_path_heat_cap}/heat_cap_stim2.png')

                mag_spin_stim0.savefig(f'{save_path_mag_spins}/mag_spins_stim0.png')
                mag_spin_stim1.savefig(f'{save_path_mag_spins}/mag_spins_stim1.png')
                mag_spin_stim2.savefig(f'{save_path_mag_spins}/mag_spins_stim2.png')

                energy_temp_stim0.savefig(f'{save_path_energy}/energy_temp_stim0.png')
                energy_temp_stim1.savefig(f'{save_path_energy}/energy_temp_stim1.png')
                energy_temp_stim2.savefig(f'{save_path_energy}/energy_temp_stim2.png')

                plt.close(en_pos_plot_stim0)
                plt.close(en_pos_plot_stim1)
                plt.close(en_pos_plot_stim2)

                plt.close(heat_cap_stim0)
                plt.close(heat_cap_stim1)
                plt.close(heat_cap_stim2)

                plt.close(mag_spin_stim0)
                plt.close(mag_spin_stim1)
                plt.close(mag_spin_stim2)

                plt.close(energy_temp_stim0)
                plt.close(energy_temp_stim1)
                plt.close(energy_temp_stim2)