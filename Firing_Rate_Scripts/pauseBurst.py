from utility import *

"""
critical behavior relation to firing rate

The pauser-burster behavior changes between reaches for the purkinje cells

Right now classifies cells as pausers or bursters based on the slope of a linear regression performed between
the x-position and the firing rate. Stores the classification with the corresponding multiplier value for that cell
based on the model generated for that session.
"""
# all print statements exist to check that the directory indices are aligned properly
# directory for all six mice in the data set
mice = os.listdir(f'./Data')
print(mice)
for mouse in mice:
    # directory for all the groups of cells ranging from All Cells, Non-Purkinje Cells, and Purkinje Cells
    cells = os.listdir(f'./Data/{mouse}')
    print(cells)
    for cell in cells:
        # all data was taken from a bin size of 10 in the matlab files and models constructed from this
        bin_data = os.listdir(f'./Data/{mouse}/{cell}')
        print(bin_data)

        data_sessions = sorted(os.listdir(f'./Data/{mouse}/{cell}/{bin_data[0]}'), key=sort_files)
        print(data_sessions)

        model_sessions = sorted(os.listdir(f'./Output/Model/{mouse}/{cell}'), key=sort_files)
        print(model_sessions)

        # check that there are an equal number of data sessions as there are model sessions
        # every data session should have a corresponding model for that session
        assert len(data_sessions) == len(model_sessions)

        for i in range(len(data_sessions)):
            # Load the data file
            base_path = f'./Data/{mouse}/{cell}/{bin_data[0]}/{data_sessions[i]}'
            b = loadmat(base_path)

            neural_spikes, continuous_kin = load_data(b)
            num_of_cells = neural_spikes[0][0].shape[1]

            model_base_path = f'./Output/Model/{mouse}/{cell}/{model_sessions[i]}'

            stim_fr_dict = {}
            avg_firing_rate = []
            avg_position = []

            # Gets an average firing rate for each cell in each individual reach
            # for stim in range(len(neural_spikes)):
            #     spike_fire_dict = {}
            #     for reach in range(len(neural_spikes[stim])):
            #         spike_fire_dict.update({f'reach{reach+1}': rolling_average_firing_rate_2d(neural_spikes[stim][reach], 50)})

            # Gets an average firing rate for each cell that is based on the average across all reaches
            # since the model is generated as an average across all reaches this is used for comparison to the model
            # the average firing rate is layered such that the first index is the stim and the second is the time slice and cell
            # example: avg_firing_rate[0][50:, 0] gives the average firing rate for stim 0 starting at time index 50 of cell 0
            for stim in range(len(neural_spikes)):
                avg_firing_rate += [rolling_average_firing_rate_2d_per_cell_all_reach(neural_spikes[stim], 50)]

            for stim in range(len(continuous_kin)):
                avg_position += [average_position(continuous_kin[stim])]

            for stim in range(len(avg_firing_rate)):
                pb_output_path = f'./Output/Stats/pb/{mouse}/{model_sessions[i]}/{cell}/stim{stim}'

                if not os.path.exists(pb_output_path):
                    os.makedirs(pb_output_path)

                pb_dict = {}

                # change the .npy file to the respective model used for comparison; would need to modify time slices of
                # data for firing rate as well
                # pre-reach is up to index 451 exclusive
                # active reach is from 451 to the end
                # mid_peak_reach is from 451 inclusive to 601 exclusive
                reach_mult = np.load(f'{model_base_path}/stim{stim}/full_reach.npy')

                pb_list = []

                # If slope is less than 0 then it is classified as a pauser; greater than 0 as a burster
                # linear regression between the x-position and the firing rate; truncated first 50 indices due to 0 firing
                for cell in range(avg_firing_rate[stim].shape[1]):
                # 0 = slope, 1 = intercept, 2 = r, 3 = p, 4 = std_err
                    fire_rate_lr = stats.linregress(avg_position[stim][50:, 0], avg_firing_rate[stim][50:, cell])
                    if fire_rate_lr[0] < 0:
                        pb_list += [(cell+1, 'pauser', f'{reach_mult[cell]}')]
                    elif fire_rate_lr[0] > 0:
                        pb_list += [(cell+1, 'burster', f'{reach_mult[cell]}')]

                pb_frame = pd.DataFrame(pb_list, columns=['cell', 'p or b', 'h-val'])
                pb_frame.to_csv(f'{pb_output_path}/all_reaches_stim{stim}_pb.csv')
