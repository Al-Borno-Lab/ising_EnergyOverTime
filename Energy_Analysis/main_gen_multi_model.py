import argparse

from utility import *

"""
main script to run calculations to find the ising model multipliers for the desired reach duration
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--DATA_LOAD_FILE_DIR",
                        default='./Data',
                        help="Directory for loading data files")

    parser.add_argument("--MODEL_SAVE_FILE_DIR",
                        default='./Output/Model',
                        help="Directory for saving output files")

    parser.add_argument("--JOB_NUM",
                       default='1',
                       help="The job number (iteration) for the script")

    parser.add_argument("--MOUSE",
                        default='cChR1',
                        help="Which mouse data to use")

    parser.add_argument("--TIME_SEG",
                        default='full',
                        help="Which time segment of the reach to use for generating the model. Default is full."
                             "pre will use time segment up to 451. active will use time segment from 451 to end."
                             "mid will use time segment starting at 451 up to 601.")

    args = parser.parse_args()

    data_file_path = f'{args.DATA_LOAD_FILE_DIR}/{args.MOUSE}/P_Cells/bin10_data'
    data_files = sorted(os.listdir(data_file_path), key=sort_files)
    BIN_SIZE = 10

    for file in data_files:
        mat_data = loadmat(f'{data_file_path}/{file}')
        spike_data, kin_data = load_data(mat_data)
        curr_file = f'{data_file_path}/{file}'
        N = spike_data[0][0].shape[1]

        if N >= 20:
            print(f'Calculating multipliers for {N} P Cells of {args.MOUSE} day {file[0:4]} job {args.JOB_NUM}\n')
        else:
            continue

        if args.TIME_SEG == 'pre':
            print("Using pre-reach time segment\n")
            save_path = f'{args.MODEL_SAVE_FILE_DIR}/{args.MOUSE}/pre_reach/session_{file[0:4]}_{N}_P_Cells/stim0/job_{args.JOB_NUM}'

            for reach in range(len(spike_data[0])):
                spike_data[0][reach] = spike_data[0][reach][:451, :]
        elif args.TIME_SEG == 'active':
            save_path = f'{args.MODEL_SAVE_FILE_DIR}/{args.MOUSE}/active_reach/session_{file[0:4]}_{N}_P_Cells/stim0/job_{args.JOB_NUM}'

            for reach in range(len(spike_data[0])):
                spike_data[0][reach] = spike_data[0][reach][451:, :]
        elif args.TIME_SEG == 'mid':
            save_path = f'{args.MODEL_SAVE_FILE_DIR}/{args.MOUSE}/mid_reach/session_{file[0:4]}_{N}_P_Cells/stim0/job_{args.JOB_NUM}'

            for reach in range(len(spike_data[0])):
                spike_data[0][reach] = spike_data[0][reach][451:601, :]
        else:
            save_path = f'{args.MODEL_SAVE_FILE_DIR}/{args.MOUSE}/full_reach/session_{file[0:4]}_{N}_P_Cells/stim0/job_{args.JOB_NUM}'

        cat_spikes = np.vstack(spike_data[0])

        bin_cat_p = preprocessingSpikes(cat_spikes, BIN_SIZE)
        bin_cat_p = 2 * bin_cat_p - 1

        stim_multipliers = ising_model_multipliers(bin_cat_p, N, save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(f'{save_path}/{args.TIME_SEG}_reach_{args.JOB_NUM}.npy', stim_multipliers)



