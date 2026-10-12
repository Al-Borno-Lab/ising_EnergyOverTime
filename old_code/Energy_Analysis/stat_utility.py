import matplotlib.pyplot as plt

from modules import *

@njit(cache=True)
def fast_sum(J, s):
    e = np.zeros(s.shape[0])
    for n in range(s.shape[0]):
        k = 0
        for i in range(s.shape[1]-1):
            for j in range(i+1, s.shape[1]):
                e[n] += J[k]*s[n,i]*s[n,j]
                k += 1
    return e

@njit("float64[:](int64[:,:],float64[:])")
def calc_e(s, params):
    e = -fast_sum(params[s.shape[1]:], s)
    e -= np.sum(s*params[:s.shape[1]], 1)
    return e

    return calc_e

def position_time(kin_stim: np.ndarray):
    time_idx_stim0 = np.min([len(kin_stim[0][i]) for i in range(0, len(kin_stim[0]))])
    time_idx_stim1 = np.min([len(kin_stim[1][i]) for i in range(0, len(kin_stim[1]))])
    time_idx_stim2 = np.min([len(kin_stim[2][i]) for i in range(0, len(kin_stim[2]))])

    x_stim_0 = np.array([kin_stim[0][i][:time_idx_stim0, 0] for i in range(0, len(kin_stim[0]))])
    x_stim_1 = np.array([kin_stim[1][i][:time_idx_stim1, 0] for i in range(0, len(kin_stim[1]))])
    x_stim_2 = np.array([kin_stim[2][i][:time_idx_stim2, 0] for i in range(0, len(kin_stim[2]))])

    return x_stim_0, x_stim_1, x_stim_2

def energy_time(neural_stim: np.ndarray, multipliers: list):
    time_idx_stim0 = np.min([len(neural_stim[0][i]) for i in range(0, len(neural_stim[0]))])
    time_idx_stim1 = np.min([len(neural_stim[1][i]) for i in range(0, len(neural_stim[1]))])
    time_idx_stim2 = np.min([len(neural_stim[2][i]) for i in range(0, len(neural_stim[2]))])

    neural_0 = (np.asarray([neural_stim[0][i][:time_idx_stim0,:] for i in range(0, len(neural_stim[0]))]) > 0)*1
    neural_1 = (np.asarray([neural_stim[1][i][:time_idx_stim1,:] for i in range(0, len(neural_stim[1]))]) > 0)*1
    neural_2 = (np.asarray([neural_stim[2][i][:time_idx_stim2,:] for i in range(0, len(neural_stim[2]))]) > 0)*1

    e_0 = np.asarray([calc_e(i, multipliers) for i in neural_0])
    e_1 = np.asarray([calc_e(i, multipliers) for i in neural_1])
    e_2 = np.asarray([calc_e(i, multipliers) for i in neural_2])

    return e_0, e_1, e_2

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_position(x_stim):
    mean_x, lower_bound, upper_bound = [], [], []
    for i in range(x_stim.shape[1]):
        m, ml, mu = mean_confidence_interval(x_stim[:, i])
        mean_x.append(m)
        lower_bound.append(ml)
        upper_bound.append(mu)

    fig, ax = plt.subplots()
    ax.set_title("Distribution of X Component over time")
    ax.plot(mean_x, '-r', label='mean')
    ax.plot(lower_bound, '-b', label='lower', alpha=0.15)
    ax.plot(upper_bound, 'b', label='upper', alpha=0.15)
    plt.fill_between(list(range(len(mean_x))), upper_bound, lower_bound, color="k", alpha=0.15)

    return fig, ax

def plot_energy_time(stim_energy):
    mean_energy, lower_bound, upper_bound = [], [], []
    for i in range(stim_energy.shape[1]):
        m, ml, mu = mean_confidence_interval(stim_energy[:, i])
        mean_energy.append(m)
        lower_bound.append(ml)
        upper_bound.append(mu)


    fig, ax = plt.subplots()
    ax.set_title("Distribution of Energy over time")
    ax.plot(mean_energy, '-r', label='mean')
    ax.plot(lower_bound, '-b', label='lower', alpha=0.15)
    ax.plot(upper_bound, 'b', label='upper', alpha=0.15)
    ax.fill_between(list(range(len(mean_energy))), upper_bound, lower_bound, color="k", alpha=0.15)
    return fig, ax

def combine_energy_position(stim_energy, stim_position, f_spline, crit_temp, energy_c, temp_c):
    mean_energy, lower_bound_en, upper_bound_en = [], [], []

    for i in range(stim_energy.shape[1]):
        m, ml, mu = mean_confidence_interval(stim_energy[:, i])
        mean_energy.append(m)
        lower_bound_en.append(ml)
        upper_bound_en.append(mu)

    mean_x, lower_bound_x, upper_bound_x = [], [], []
    for i in range(stim_position.shape[1]):
        m, ml, mu = mean_confidence_interval(stim_position[:, i])
        mean_x.append(m)
        lower_bound_x.append(ml)
        upper_bound_x.append(mu)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,15))
    axes[0].set_title("Distribution of Energy over time")
    fig.text(0, 0.95, f'Crit Energy: {energy_c}\nCrit Temp: {temp_c}')
    fig.text(0, 0.9, f'Crit Energy Spline: {f_spline(crit_temp)}\nCrit Temp: {crit_temp}')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Energy")
    axes[0].plot(mean_energy, '-r', label='mean')
    axes[0].plot(lower_bound_en, '-b', label='lower', alpha=0.15)
    axes[0].plot(upper_bound_en, 'b', label='upper', alpha=0.15)
    axes[0].fill_between(list(range(len(mean_energy))), upper_bound_en, lower_bound_en, color="k", alpha=0.15)
    axes[0].plot([0, 950], [f_spline(crit_temp), f_spline(crit_temp)], label='spline-energy')
    axes[0].plot([0, 950], [energy_c, energy_c], label='crit-energy')
    axes[0].legend()

    axes[1].set_title("Distribution of X Component over time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("X-Position")
    axes[1].plot(mean_x, '-r', label='mean')
    axes[1].plot(lower_bound_x, '-b', label='lower', alpha=0.15)
    axes[1].plot(upper_bound_x, 'b', label='upper', alpha=0.15)
    axes[1].fill_between(list(range(len(mean_x))), upper_bound_x, lower_bound_x, color="k", alpha=0.15)
    axes[1].legend()

    return fig

def metropolis(initial_v, multiplier, temp, bootStrap=1000, samples=100000):
    net_spin = []
    energy_spin = []

    current_vec = initial_v.copy()

    for i in range(0, 100000):
        E_i = calc_e(current_vec.reshape(1, -1), multiplier)[0]

        # permutate vector
        index = np.random.randint(0, high=current_vec.shape[1])
        mu_vector = current_vec.copy()
        mu_vector[:, index] *= -1

        E_u = calc_e(mu_vector.reshape(1, -1), multiplier)[0]

        # accept or reject altered vector

        dE = E_u - E_i
        # same thing?
        if (dE > 0) * (np.random.random() < np.exp(-temp * dE)):
            current_vec = mu_vector
        elif dE <= 0:
            current_vec = mu_vector

        if i > 10000 - bootStrap:
            net_spin += [current_vec.sum()]
            energy_spin += [calc_e(current_vec.reshape(1, -1), multiplier)[0]]

    return current_vec, net_spin, energy_spin

def phase_graph(spins: np.ndarray, multipliers: list, n: int):
    avg_spin = []

    avg_energy_c = []

    temperature_range = []
    spins = spins.reshape(1, n)
    temp = 0.1
    while temp < 3:
        print(f"temp: {temp}")
        temperature_range.append(temp)
        _, net_spin, net_energy = metropolis(spins, multipliers, 1/temp)

        if temp < 0.7:
            temp += 0.1
        elif temp < 1.7:
            temp += 0.01
        else:
            temp += 0.1

        avg_spin += [net_spin]

        avg_energy_c += [net_energy]

    return avg_energy_c, avg_spin, temperature_range


def system_size_plots(save_dir: str, system_spins: list[np.ndarray], system_multipliers: list[list], num_of_cells: list[int]):
    temperature_range = np.arange(0.1, 3, 0.05)
    heat_capacity = lambda x: np.power(np.asarray(x), 2).mean() - np.power(np.asarray(x).mean(), 2)

    # average heat capacity for the whole day
    avg_system_heat_capacity = []
    avg_system_energy = []
    avg_system_magnetization = []

    # average heat capacity for each stim in a given day
    for this_day in range(len(system_spins)):
        avg_heat_capacity = []
        avg_energy = []
        avg_spin = []

        for stim in range(len(system_spins[this_day])):
            spins = system_spins[this_day][stim][0, :]
            spins = spins.reshape(1, num_of_cells[this_day])
            multipliers = system_multipliers[this_day][stim]

            avg_pos = []
            avg_energy_c = []
            print(f"Calculating for day {this_day+1} stim{stim}")
            for temp in temperature_range:
                print(f"temp: {temp}")

                _, net_spin, net_energy = metropolis(spins, multipliers, 1 / temp)

                avg_pos += [net_spin]

                avg_energy_c += [net_energy]


            avg_spin += [[np.mean(r) / num_of_cells[this_day] ** 2 for r in avg_pos]]

            avg_energy += [[np.mean(r) for r in avg_energy_c]]

            avg_heat_capacity += [[heat_capacity(r) for r in avg_energy_c]]

        avg_system_heat_capacity += [avg_heat_capacity]
        avg_system_energy += [avg_energy]
        avg_system_magnetization += [avg_spin]

    fig_save_dir = save_dir

    for this_stim in range(len(avg_system_heat_capacity[0])):
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        fig3, ax3 = plt.subplots(nrows=1, ncols=1)
        fig_save_dir = f'{save_dir}/system_size_comparison/stim{this_stim}'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)

        for this_day in range(len(avg_system_heat_capacity)):
            Tc_idx = np.argmax(avg_system_heat_capacity[this_day][this_stim])

            c_temp = temperature_range[Tc_idx]

            ax1.set_title(f'Heat Capacity for Stim{this_stim}')
            ax1.set_xlabel('Temperature')
            ax1.set_ylabel('Heat Capacity')
            ax1.plot(temperature_range, avg_system_heat_capacity[this_day][this_stim], label=f'{num_of_cells[this_day]} cells')
            ax1.plot([c_temp, c_temp], [0, max(avg_system_heat_capacity[this_day][this_stim])])
            ax1.legend(loc='best')

            ax2.set_title('Average Spin per Temperature')
            ax2.set_xlabel('Temperature')
            ax2.set_ylabel('Average Spin')
            ax2.plot(temperature_range, avg_system_magnetization[this_day][this_stim], label=f'{num_of_cells[this_day]}')

            conv_bar = np.mean(avg_system_magnetization[this_day][this_stim][-10:])
            ax2.plot([temperature_range[0], temperature_range[-1]], [conv_bar, conv_bar])
            ax2.legend(loc='best')

            critical_energy = avg_system_energy[this_day][this_stim][Tc_idx]
            c_temp = temperature_range[Tc_idx]

            ax3.set_title('Energy per Temperature')
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Average Energy')
            ax3.plot(temperature_range, avg_system_energy[this_day][this_stim], label=f'{num_of_cells[this_day]}')
            ax3.plot([c_temp, c_temp], [min(avg_system_energy[this_day][this_stim]), max(avg_system_energy[this_day][this_stim])])
            ax3.legend(loc='best')

        fig1.savefig(f'{fig_save_dir}/heat_capacity.png')
        fig2.savefig(f'{fig_save_dir}/avg_energy.png')
        fig3.savefig(f'{fig_save_dir}/avg_spin.png')

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def plot_magnetization(spins: list, temperature_range: np.ndarray):

    conv_bar = np.mean(spins[-10:])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    ax.plot([temperature_range[0], temperature_range[-1]], [conv_bar, conv_bar])

    ax.set_title("Avg Spin per Temperature")
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Average Spin')
    ax.plot(temperature_range, spins, c='blue', label='full-reach')

    return fig


def plot_heat_capacity(avg_energy: list, temperature_range: np.ndarray):
    heat_capacity = lambda x: np.power(np.asarray(x), 2).mean() - np.power(np.asarray(x).mean(), 2)
    avg_heat_capacity_full = [heat_capacity(r) for r in avg_energy]
    # avg_heat_capacity_pre = [heat_capacity(r) for r in avg_energy[:451]]
    # avg_heat_capacity_active = [heat_capacity(r) for r in avg_energy[451:]]

    Tc_idx_full = np.argmax(avg_heat_capacity_full)
    # Tc_idx_pre = np.argmax(avg_heat_capacity_pre)
    # Tc_idx_active = np.argmax(avg_heat_capacity_active)

    c_temp_full = temperature_range[Tc_idx_full]
    # c_temp_pre = temperature_range[Tc_idx_pre]
    # c_temp_active = temperature_range[Tc_idx_active]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))
    fig.text(0, 0.05, f"Critical Temp: {c_temp_full}")
    ax.set_title('Heat Capacity')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Heat Capacity')
    ax.plot(temperature_range, avg_heat_capacity_full, c='blue', label='full-reach')
    ax.plot([c_temp_full, c_temp_full], [0, max(avg_heat_capacity_full)])

    # ax[0,0].set_title('Heat Capacity')
    # ax[0,0].set_xlabel('Temperature')
    # ax[0,0].set_ylabel('Heat Capacity')
    # ax[0,0].plot(temperature_range, avg_heat_capacity_pre, c='blue', label='pre-reach')
    # ax[0,0].plot([c_temp_pre, c_temp_pre], [0, max(avg_heat_capacity_pre)])

    # ax[0,1].set_title('Heat Capacity')
    # ax[0,1].set_xlabel('Temperature')
    # ax[0,1].set_ylabel('Heat Capacity')
    # ax[0,1].plot(temperature_range, avg_heat_capacity_active, c='blue', label='active-reach')
    # ax[0,1].plot([c_temp_active, c_temp_active], [0, max(avg_heat_capacity_active)])

    return fig, Tc_idx_full, c_temp_full

def plot_energy(energy: list, temperature_range: np.ndarray, crit_temp_idx: int64):
    # avg_energy = [np.mean(r) for r in energy]

    critical_energy = energy[crit_temp_idx]
    c_temp = temperature_range[crit_temp_idx]

    # print(f"Critical temperature: {c_temp}")
    # print(f"Critical Energy: {critical_energy}")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))
    fig.text(0, 0.05, f'Critical temperature: {c_temp}\nCritical Energy:{critical_energy}')
    ax.set_title('Energy per Temperature')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Average Energy')
    ax.plot(temperature_range, energy, c='blue', label='full-reach')
    ax.plot([c_temp, c_temp], [min(energy), max(energy)])
    ax.legend()
    # plt.savefig(f'{save_dir}/energy.png')
    # plt.close()
    return fig, critical_energy, c_temp