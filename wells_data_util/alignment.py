import json
import csv

def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    nbr_reaches = len(data['xpos_toEnd'])
    folder = 'paw_positions/'

    for i in range(0,1):
        xpos_data = data['xpos_toEnd']['0']
        ypos_data = data['ypos_toEnd']['0']

        filename_xpos = folder + 'xpos_data' + str(i) + '.csv'
        filename_ypos = folder + 'ypos_data' + str(i) + '.csv'

        with open(filename_xpos, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(map(lambda x: [x], xpos_data))

        with open(filename_ypos, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(map(lambda x: [x], ypos_data))

def read_neuralData_json(filename, mouse_id, reach_id):
    with open(filename, 'r') as file:
        data = json.load(file)
    mouse_dict = data['mouse']
    mouse_list = [k for k, v in mouse_dict.items() if v == mouse_id]
    spike_times = [data['times'][x] for x in mouse_list]

    with open('data/wells_data/df_reaches.json', 'r') as file:
        reach_data = json.load(file)

    reachMax_ephys = reach_data['reachMax_ephys'][reach_id]

    reach_max_idx = reach_data['reachMax_ind'][reach_id]
    time_prior_reach_max = reach_max_idx / 150.0
    reach_len = len(reach_data['xpos_toEnd'][reach_id])
    time_after_reach_max = (reach_len - reach_max_idx) / 150
    first_time = reachMax_ephys - time_prior_reach_max - 0.6
    last_time = reachMax_ephys  + 0.4

    pop_spikes = []
    for idx, elm in enumerate(spike_times):

        first_idx = closest(spike_times[idx], first_time)
        second_idx = closest(spike_times[idx], last_time)

        spikes = spike_times[idx][first_idx:second_idx+1]
        pop_spikes.append(spikes)

    print(pop_spikes)

def closest(lst, K):
     return lst.index(lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))])

if __name__ == '__main__':
    #read_json('/Users/mazen/Documents/WelleLabData/df_reaches.json')
    mouse_id = 103
    reach_id = '0'
    read_neuralData_json('data/wells_data/df_neural.json', mouse_id, reach_id)


