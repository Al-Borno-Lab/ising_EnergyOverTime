from modules import *

def tryint(s):
    try:
        return int(s)
    except:
        return s

def sort_files(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def preprocessingSpikes(spikes, binSize):
    convolved_spikes = 1 * (np.lib.stride_tricks.sliding_window_view(spikes, binSize, axis=0).sum(axis=2) > 0)

    return convolved_spikes

def checkPerturbationRegime(spikes, binSize):

    # Run convolution
    spike_bins = preprocessingSpikes(spikes, binSize)

    # Count average spike per neuron & average of expect neuron fire rate
    avgSpikePerNeuron = np.sum(spike_bins, axis=0) / spike_bins.shape[0]
    avgExpectedNeuronFire = np.sum(avgSpikePerNeuron) / spike_bins.shape[1]

    # Return N_c
    N_c = 1 / (avgExpectedNeuronFire * binSize)
    return N_c

def binSizeScan(spikes, start=1, end=200):

    # Scan over bin sizes
    N_cList = []
    for binSize in range(start, end):
        N_cList += [checkPerturbationRegime(spikes, binSize)]

    return np.asarray(N_cList)

def identifyReactiveTScore(neural_stim, continuous_stim, window_size):
    stim_0_firing_rate = []
    stim_1_firing_rate = []

    # Finding the 50ms window in stim 0 containing the 1 cm x-position
    # and computing the sliding window firing rates
    for reach_i in range(len(neural_stim[0])):

        try:
            neural_0_i = neural_stim[0][reach_i]
            continues_0_i = continuous_stim[0][reach_i][:, 0]

            # Storing index of max reach velocity
            # idx_max_v = np.argmax(continues_0_i[:, 3])
            # print(idx_max_v)
            idx_x_1cm = next(x for x, val in enumerate(continues_0_i) if val > 0.99)

            # Identifying 50ms window containing max reach velocity
            # ms50_window = neural_0_i[idx_max_v: idx_max_v + 50]
            ms50_window = neural_0_i[idx_x_1cm:idx_x_1cm + 50]
            # Storing average firing rates within a 15ms sliding window
            stim_0_firing_rate.append(rolling_average_firing_rate_2d(ms50_window, window_size))
        except:
            print(reach_i, "Failed")

    stim_0_firing_rate = np.vstack(stim_0_firing_rate)

    # Finding the 50ms window in stim 1 containing the perturbation and
    # computing the sliding window firing rates
    for reach_i in range(len(neural_stim[1])):

        try:
            neural_1_i = neural_stim[1][reach_i]
            continues_1_i = continuous_stim[1][reach_i][:, 0]

            # Storing index of max reach velocity
            # idx_max_v = np.argmax(continues_1_i[:, 3])
            # print(idx_max_v)
            idx_x_1cm = next(x for x, val in enumerate(continues_1_i) if val > 0.99)
            # Identifying 50ms window containing perturbation
            ms50_window = neural_1_i[idx_x_1cm : idx_x_1cm + 50]

            # Storing average firing rates within a 15ms sliding window
            stim_1_firing_rate.append(rolling_average_firing_rate_2d(ms50_window, window_size))
        except:
            print(reach_i, "Failed")

    stim_1_firing_rate = np.vstack(stim_1_firing_rate)

    n_test = stats.ttest_ind(stim_0_firing_rate, stim_1_firing_rate, axis=0)

    return n_test.pvalue

def average_position(position_matrix):
    min_time_points = min(matrix.shape[0] for matrix in position_matrix)

    truncated_matrices = [matrix[:min_time_points, :] for matrix in position_matrix]

    avg_position = np.mean(np.stack(truncated_matrices), axis=0)

    return avg_position

def rolling_average_firing_rate_2d_per_cell_all_reach(spike_matrix, window_size):
    reaches = len(spike_matrix)
    time, neurons = spike_matrix[0].shape
    min_time_points = min(matrix.shape[0] for matrix in spike_matrix)
    firing_rate_matrices = []

    for reach in range(reaches):
        firing_rate_matrix = np.zeros((min_time_points, neurons), dtype=float)

        for neuron in range(neurons):
            spike_vector = spike_matrix[reach][:min_time_points, neuron]
            cumsum = np.cumsum(np.insert(spike_vector, 0, 0))
            rolling_sum = cumsum[window_size:] - cumsum[:-window_size]
            firing_rate = rolling_sum / window_size

            firing_rate_matrix[:, neuron] = np.concatenate((np.zeros(window_size - 1), firing_rate))

        firing_rate_matrices.append(firing_rate_matrix)

    avg_firing_rate = np.mean(np.stack(firing_rate_matrices), axis=0)  # Use np.stack to enforce uniform shape

    return avg_firing_rate

def rolling_average_firing_rate_2d(spike_matrix, window_size):
  num_neurons = spike_matrix.shape[1]
  firing_rate_matrix = np.zeros_like(spike_matrix, dtype=float)

  for neuron in range(num_neurons):
    spike_vector = spike_matrix[:, neuron]
    cumsum = np.cumsum(np.insert(spike_vector, 0, 0))
    rolling_sum = cumsum[window_size:] - cumsum[:-window_size]
    firing_rate = rolling_sum / window_size
    firing_rate_matrix[:, neuron] = np.concatenate((np.zeros(window_size - 1), firing_rate))

  return firing_rate_matrix

def Q_distribution(neuronsSpikeSet, numBins=100, label=""):
    '''
        Enter in spike data, could be sampled from model or empirical
        Function will output a histogram of average number of neurons that participated in a time frame
    '''

    # Initial params
    numN = neuronsSpikeSet.shape[1]
    Q_samples = []
    neuronsSpikeSet = neuronsSpikeSet.copy()

    # check data is on in [0, 1] scale, not [-1, 1]
    if np.min(neuronsSpikeSet) == -1:
        neuronsSpikeSet[neuronsSpikeSet == -1] = 0

    for n in neuronsSpikeSet:

        # Fraction of neurons that fired
        Q = np.sum(n) / (numN * 2.) + 1/2.
        Q_samples += [Q]

    # Calculate weight
    Q_samples = np.asarray(Q_samples)
    weight = np.ones_like(Q_samples) / np.prod(Q_samples.shape)

    # Plot distributions
    return plt.hist(Q_samples, bins=numBins, weights=weight, alpha=0.5, label=label, density=True)

def loadmat(filename):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def velocity_to_acceleration(timesteps, velocity):

    delta_t = timesteps[1] - timesteps[0]

    return np.gradient(velocity, delta_t)

def load_data(b):
    neural_stim = []
    continuous_stim = []

    for j in b['kinAggrogate'].keys():

        stim = b['kinAggrogate'][j]

        neural_session = []
        continuous_sessions = []

        for k in stim.keys():
            dataMatrix = np.array(stim[k])

            if k[0] == "n":
                neural_session += [dataMatrix[:, 1:]]

            if k[0:2] == "l_":
                ax = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, -3]).reshape(-1, 1)
                ay = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, -2]).reshape(-1, 1)
                az = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, -1]).reshape(-1, 1)

                # print(dataMatrix.shape, ax.shape)
                dataMatrix = np.hstack([dataMatrix, ax, ay, az])

                jx = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, 7]).reshape(-1, 1)
                jy = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, 8]).reshape(-1, 1)
                jz = velocity_to_acceleration(dataMatrix[:, 0], dataMatrix[:, 9]).reshape(-1, 1)

                dataMatrix = np.hstack([dataMatrix, jx, jy, jz])
                continuous_sessions += [dataMatrix[:, 1:]]

        neural_stim += [neural_session]
        continuous_stim += [continuous_sessions]

    return neural_stim, continuous_stim

def ising_model_multipliers(spikes: np.ndarray,
                            save_dir: str):
    BIN_SIZE = 1
    SAMPLE_SIZE = 100000
    N_CPUS = os.cpu_count()
    MAX_ITER = 75
    ETA = 1e-3

    stim_0 = spikes

    # Concatenate all reaches together
    bin_cat = np.vstack(stim_0)

    bin_cat_p = preprocessingSpikes(bin_cat, BIN_SIZE)
    bin_cat_p = 2 * bin_cat_p - 1

    N = bin_cat_p.shape[1]

    solver = MCH(bin_cat_p,
                 sample_size=SAMPLE_SIZE,
                 rng=np.random.RandomState(0),
                 n_cpus=N_CPUS,
                 sampler_kw={'boost':True})

    def learn_settings(i):
        return {'maxdlamda': 1, 'eta': ETA}

    stim_0_multipliers = solver.solve(maxiter=MAX_ITER,
                                      n_iters=N*10,
                                      burn_in=N*10,
                                      custom_convergence_f=learn_settings)

    return stim_0_multipliers

def verify_fit(spikes):
    BIN_SIZE = 1
    SAMPLE_SIZE = 100000
    N_CPUS = 20
    MAX_ITER = 75
    ETA = 1e-3

    stim_0 = spikes[0]

    # Concatenate all reaches together
    bin_cat = np.vstack(stim_0)

    bin_cat_p = preprocessingSpikes(bin_cat, BIN_SIZE)
    bin_cat_p = 2 * bin_cat_p - 1

    N = bin_cat_p.shape[1]
    solver = MCH(bin_cat_p,
                 sample_size=SAMPLE_SIZE,
                 rng=np.random.RandomState(0),
                 n_cpus=N_CPUS)

    def learn_settings(i):
        return {'maxdlamda': 1, 'eta': ETA}

    stim_0_multipliers = solver.solve(maxiter=MAX_ITER,
                                      n_iters=N * 10,
                                      burn_in=N * 10,
                                      iprint='detailed',
                                      custom_convergence_f=learn_settings)

    for i in range(3, 6):
        plt.title(f'{i}-corr')
        plt.xlabel('Empirical Data Correlations')
        plt.ylabel('Sample Data Correlations')
        plt.scatter(k_corr(bin_cat_p, i), k_corr(solver.model.sample, i))
        plt.plot([0, 1], [0, 1])
        plt.show()

