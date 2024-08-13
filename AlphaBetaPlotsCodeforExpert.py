import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

class Simulation:
    def __init__(self, start_trial, end_trial, Q_i, df):
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.Num_trials = end_trial - start_trial
        self.Q_i = Q_i
        self.R = df['rew_t'].iloc[start_trial:end_trial].reset_index(drop=True).astype(int)
        self.states = df['tone_freq'].iloc[start_trial:end_trial].replace({6000: '6kHz', 10000: '10kHz'}).reset_index(drop=True)
        self.actions = df['response'].iloc[start_trial:end_trial].reset_index(drop=True)
        self.flags = df['expert'].iloc[start_trial:end_trial].reset_index(drop=True)

    def Delta(self, A):
        N = len(self.states)
        Q = self.Q_i.copy()
        for t in range(N):
            state, action = self.states[t], self.actions[t]
            Q[state][action] += A * (self.R[t] - Q[state][action])
        return Q

    def log(self, A, B):
        log_likelihood = []
        Q = {'6kHz': {'L': np.zeros(self.Num_trials), 'R': np.zeros(self.Num_trials), 'N': np.zeros(self.Num_trials)},
             '10kHz': {'L': np.zeros(self.Num_trials), 'R': np.zeros(self.Num_trials), 'N': np.zeros(self.Num_trials)}}

        for t in range(len(self.states)):
            state = self.states[t]
            action = self.actions[t]
            reward = self.R[t]

            total_exp = np.exp(B * Q[state]['L'][t]) + np.exp(B * Q[state]['R'][t]) + np.exp(B * Q[state]['N'][t])
            P_L = np.exp(B * Q[state]['L'][t]) / total_exp
            P_R = np.exp(B * Q[state]['R'][t]) / total_exp
            P_N = 1 - (P_L + P_R)

            P_t = {'L': P_L, 'R': P_R, 'N': P_N}[action]
            log_likelihood.append(np.log(P_t) if P_t > 1e-15 else -1e15)

            if t + 1 < self.Num_trials:
                for a in ['L', 'R', 'N']:
                    Q[state][a][t+1] = Q[state][a][t] + A * (reward - Q[state][a][t]) if a == action else Q[state][a][t]
        return log_likelihood

    def neg_log_likelihood(self, params):
        A, B = params
        return -1 * np.sum(self.log(A, B))

class MLE:
    def __init__(self, start_trial, end_trial, Q_i, df):
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.Q_i = Q_i
        self.df = df
        self.A_range = np.linspace(10**(-50), 0.99, 50)  # Reduced grid size for faster computation
        self.B_range = np.linspace(1, 30, 50)

    def evaluate_params(self, params):
        A, B = params
        real = Simulation(self.start_trial, self.end_trial, self.Q_i, self.df)
        return real.neg_log_likelihood([A, B])

    def func(self):
        param_pairs = [(A, B) for A in self.A_range for B in self.B_range]
        with Pool(cpu_count()) as pool:
            nlog_likelihoods = pool.map(self.evaluate_params, param_pairs)

        nlog_likelihoods = np.array(nlog_likelihoods).reshape(len(self.A_range), len(self.B_range))
        ll = np.exp(np.min(nlog_likelihoods) - nlog_likelihoods)
        P = ll / np.sum(ll)
        i_min, j_min = np.unravel_index(np.argmin(nlog_likelihoods), nlog_likelihoods.shape)
        return self.A_range[i_min], self.B_range[j_min], ll, P

def find_expert_segments(expert_series, rew_prob):
    expert_segments = []
    in_segment = False
    segment_start = None

    for i in range(len(expert_series)):
        if expert_series[i] and rew_prob[i] == 0.9 and not in_segment:
            in_segment = True
            segment_start = i
        elif not (expert_series[i] and rew_prob[i] == 0.9) and in_segment:
            in_segment = False
            expert_segments.append((segment_start, i - 1))
            segment_start = None

    if in_segment:
        expert_segments.append((segment_start, len(expert_series) - 1))

    return expert_segments

def run_simulation_for_segments_rev(df):
    Q_i = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
    expert_series = df['expert'].tolist()
    rew_prob = df['rew_prob'].tolist()
    expert_segments = find_expert_segments(expert_series, rew_prob)
    start_trial = 0
    alpha_parameters = {}
    beta_parameters = {}
    if len(expert_segments) == 1:
        end_trial = expert_segments[0][0]
        for s in range(3):
            mle = MLE(start_trial, end_trial, Q_i, df)
            A_optimal, B_optimal, ll, P = mle.func()
            alpha_parameters[s] = A_optimal
            beta_parameters[s] = B_optimal
            Q_i = Simulation(start_trial, end_trial, Q_i, df).Delta(A_optimal)
            start_trial = end_trial
            if s == 0:
                end_trial = expert_segments[0][1]
            elif s == 1:
                end_trial = len(rew_prob)
    else:
        end_trial = expert_segments[0][0]
        for s in range(5):
            mle = MLE(start_trial, end_trial, Q_i, df)
            A_optimal, B_optimal, ll, P = mle.func()
            alpha_parameters[s] = A_optimal
            beta_parameters[s] = B_optimal
            Q_i = Simulation(start_trial, end_trial, Q_i, df).Delta(A_optimal)
            start_trial = end_trial
            if s == 0:
                end_trial = expert_segments[0][1]
            elif s == 1:
                end_trial = expert_segments[1][0]
            elif s == 2:
                end_trial = expert_segments[1][1]
            elif s == 3:
                end_trial = len(rew_prob)

    return alpha_parameters, beta_parameters, 2*len(expert_segments)+1

def run_simulation_for_segments_var(df):
    Q_i = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
    expert_series = df['expert'].tolist()
    rew_prob = df['rew_prob'].tolist()
    expert_segments = find_expert_segments(expert_series, rew_prob)
    start_trial = 0
    alpha_parameters = {}
    beta_parameters = {}
    end_trial = expert_segments[0][0]+250
    for s in range(2):
        mle = MLE(start_trial, end_trial, Q_i, df)
        A_optimal, B_optimal, ll, P = mle.func()
        alpha_parameters[s] = A_optimal
        beta_parameters[s] = B_optimal
        Q_i = Simulation(start_trial, end_trial, Q_i, df).Delta(A_optimal)
        start_trial = end_trial
        if s == 0:
            end_trial = expert_segments[0][1]
    return alpha_parameters, beta_parameters, 2*len(expert_segments)+1

def process_mouse_rev(link):
    df_mouse = pd.read_csv(os.path.join('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/', link))
    alpha_parameters, beta_parameters, phases = run_simulation_for_segments_rev(df_mouse)
    return link, alpha_parameters, beta_parameters, phases

def process_mouse_var(link):
    df_mouse = pd.read_csv(os.path.join('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/', link))
    alpha_parameters, beta_parameters, phases = run_simulation_for_segments_var(df_mouse)
    return link, alpha_parameters, beta_parameters, phases


def alphasbetas_rev(mouse_links):
    ALPHAS = {link: [] for link in mouse_links}
    BETAS = {link: [] for link in mouse_links}
    PHASES =[]
    # Sequentially process each mouse link to avoid nested multiprocessing
    results = [process_mouse_rev(link) for link in mouse_links]

    for link, alpha_parameters, beta_parameters, phases in results:
        ALPHAS[link].extend(alpha_parameters.values())
        BETAS[link].extend(beta_parameters.values())
        PHASES.append(phases)

    max_phases = max(PHASES)

    # Padding alpha and beta values with None or np.nan to match the max_phases
    for link in mouse_links:
        ALPHAS[link] += [np.nan] * (max_phases - len(ALPHAS[link]))
        BETAS[link] += [np.nan] * (max_phases - len(BETAS[link]))

    return ALPHAS, BETAS, max_phases

def alphasbetas_var(mouse_links):
    ALPHAS = {link: [] for link in mouse_links}
    BETAS = {link: [] for link in mouse_links}
    PHASES =[]
    # Sequentially process each mouse link to avoid nested multiprocessing
    results = [process_mouse_var(link) for link in mouse_links]

    for link, alpha_parameters, beta_parameters, phases in results:
        ALPHAS[link].extend(alpha_parameters.values())
        BETAS[link].extend(beta_parameters.values())
        PHASES.append(phases)

    max_phases = max(PHASES)

    # Padding alpha and beta values with None or np.nan to match the max_phases
    for link in mouse_links:
        ALPHAS[link] += [np.nan] * (max_phases - len(ALPHAS[link]))
        BETAS[link] += [np.nan] * (max_phases - len(BETAS[link]))

    return ALPHAS, BETAS, max_phases

if __name__ == "__main__":
    # Load data and run the simulation
    df_mice = pd.read_excel('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/Mouse_DATA.xlsx')
    # mouse_per_mouse_type = int(input('How many mouse/mouse_type would you like choose numbers ranging from 1 to 5?\n'))

    links_16p_rev = df_mice['link_16p_rev'].iloc[:].dropna().tolist()
    links_16p_var = df_mice['link_16p_var'].iloc[:].dropna().tolist()
    links_WT_rev = df_mice['link_WT_rev'].iloc[:].dropna().tolist()
    links_WT_var = df_mice['link_WT_var'].iloc[:].dropna().tolist()

    mouse_type_input = input('What mouse type would you like? (16p_rev, 16p_var, WT_rev, WT_var)\n')
    mouse_type_dict = {
        '16p_rev': links_16p_rev,
        '16p_var': links_16p_var,
        'WT_rev': links_WT_rev,
        'WT_var': links_WT_var
    }
    if mouse_type_input == '16p_rev' or mouse_type_input == 'WT_rev':
        A, B, phases = alphasbetas_rev(mouse_type_dict[mouse_type_input])
    
    else:
        A, B, phases = alphasbetas_var(mouse_type_dict[mouse_type_input])

    # Plot Alpha values
    for mouse_link in mouse_type_dict[mouse_type_input]:
        alpha_values = A[mouse_link]  
        plt.plot(range(phases), alpha_values, label=f'Alpha: {mouse_link}')

    plt.legend()
    plt.title('Alpha Parameters for ' + mouse_type_input)
    plt.xlabel('Phases')
    plt.ylabel('Alpha Value')
    plt.show()

    # Plot Beta values
    for mouse_link in mouse_type_dict[mouse_type_input]:
        beta_values = B[mouse_link]  
        plt.plot(range(phases), beta_values, label=f'Beta: {mouse_link}')

    plt.legend()
    plt.title('Beta Parameters for ' + mouse_type_input)
    plt.xlabel('Phases')
    plt.ylabel('Beta Value')
    plt.show()
