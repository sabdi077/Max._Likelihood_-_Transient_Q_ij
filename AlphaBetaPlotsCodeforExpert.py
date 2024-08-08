import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
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
        Q_history = [Q.copy()]   

        for t in range(N):
            state, action = self.states[t], self.actions[t]
            Q[state][action] = Q[state][action] + A * (self.R[t] - Q[state][action])
            Q_history.append(Q.copy())

        return Q
    def log(self, A, B):
        log_likelihood = []
        Q = {'6kHz': {'L': [0]*self.Num_trials, 'R': [0]*self.Num_trials, 'N': [0]*self.Num_trials},
             '10kHz': {'L': [0]*self.Num_trials, 'R': [0]*self.Num_trials, 'N': [0]*self.Num_trials}}

        for t in range(len(self.states)):
            state = self.states[t]
            action = self.actions[t]
            reward = self.R[t]

            # Compute the total exponential sum for normalization
            total_exp = sum(np.exp(B * Q[state][a][t]) for a in ['L', 'R', 'N'])
            P_L = np.exp(B * Q[state]['L'][t]) / total_exp
            P_R = np.exp(B * Q[state]['R'][t]) / total_exp
            P_N = 1 - (P_L + P_R)

            P_t = {'L': P_L, 'R': P_R, 'N': P_N}[action]
            log_likelihood.append(np.log(P_t) if P_t > 1e-15 else -1e15)

            if t + 1 < self.Num_trials:  # Ensure we don't go out of range
                for a in ['L', 'R', 'N']:
                    if a == action:
                        Q[state][a][t+1] = Q[state][a][t] + A * (reward - Q[state][a][t])
                    else:
                        Q[state][a][t+1] = Q[state][a][t]

        return log_likelihood

    def neg_log_likelihood(self, params):
        A, B = params
        log_likelihood = self.log(A, B)
        return -1 * np.sum(log_likelihood)

class MLE:
    def __init__(self, start_trial, end_trial, Q_i, df):
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.Q_i = Q_i
        self.df = df
        self.A_range = np.linspace(10**(-50), 0.99, 100)
        self.B_range = np.linspace(1, 30, 100)

    def func(self):
        real = Simulation(self.start_trial, self.end_trial, self.Q_i, self.df)
        nlog_likelihoods = np.zeros((len(self.A_range), len(self.B_range)))

        for i, A in enumerate(self.A_range):
            for j, B in enumerate(self.B_range):
                nlog_likelihoods[i, j] = real.neg_log_likelihood([A, B])

        ll = np.exp(np.min(nlog_likelihoods) - nlog_likelihoods)
        P = ll / np.sum(ll)
        i_min, j_min = np.unravel_index(np.argmin(ll), nlog_likelihoods.shape)
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


def run_simulation_for_segments(df):
    Q_i = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
    expert_series = df['expert'].iloc[:].reset_index(drop=True).tolist()
    rew_prob = df['rew_prob'].iloc[:].reset_index(drop=True).tolist()
    expert_segments = find_expert_segments(expert_series, rew_prob)
    results = []
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # More colors can be added if needed
    start_trial = 0
    if len(expert_segments)==1:
        end_trial=expert_segments[0]
        alpha_parameters = {a: [] for a in range(3)}
        beta_parameters = {b: [] for b in range(3)}
        for s in range(3):
            '''
            s = 0 is the pre expert phase, s=1 is the expert phase, s=2, is post expert phase
            '''
            mle = MLE(start_trial, end_trial, Q_i, df)
            A_optimal, B_optimal, ll, P, mle.func()
            alpha_parameters[s].append(A_optimal)
            beta_parameters[s].append(B_optimal)
            Q_i = Simulation(start_trial, end_trial, Q_i, df).Delta(A_optimal)
            start_trial = end_trial
            if s == 0:
                end_trial = expert_segments[1]
            elif s==1:
                end_trial = len(rew_prob)
    else:
        end_trial = expert_segments[0][0]
        alpha_parameters = {a: [] for a in range(5)}
        beta_parameters = {b: [] for b in range(5)}
        for s in range(5):
            '''
            s=0 is pre first expert phase, s=1 is first expert phase, s=2 is post first expert phase, s=3 second expert phase, s=4 is post second expert phase
            '''
            mle = MLE(start_trial, end_trial, Q_i, df)
            A_optimal, B_optimal, ll, P = mle.func()
            alpha_parameters[s].append(A_optimal)
            beta_parameters[s].append(B_optimal)
            Q_i = Simulation(start_trial, end_trial, Q_i, df).Delta(A_optimal)
            start_trial = end_trial
            if s == 0:
                end_trial = expert_segments[0][1]
            elif s == 1:
                end_trial = expert_segments[1][0]
            elif s==2:
                end_trial = expert_segments[1][1]
            elif s==3:
                end_trial = len(rew_prob)
    
    return alpha_parameters, beta_parameters

df_mice = pd.read_excel('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/Mouse_DATA.xlsx')
mouse_per_mouse_type = int(input('How many mouse/mouse_type would you like choose numbers ranging from 1 to 5?\n'))
# Ensure column names are correct
links_16p_rev = df_mice['link_16p_rev'].iloc[:mouse_per_mouse_type].dropna().tolist()
links_16p_var = df_mice['link_16p_var'].iloc[:mouse_per_mouse_type].dropna().tolist()
links_WT_rev = df_mice['link_WT_rev'].iloc[:mouse_per_mouse_type].dropna().tolist()
links_WT_var = df_mice['link_WT_var'].iloc[:mouse_per_mouse_type].dropna().tolist()

def alphasbetas(mouse_links):
    ALPHAS = {link: [] for link in mouse_links}
    BETAS = {link: [] for link in mouse_links}
    for link in mouse_links:
        df_mouse = pd.read_csv(f'/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/{link}')
        alpha_parameters, beta_parameters = run_simulation_for_segments(df_mouse)
        for i in range(len(alpha_parameters)):
            ALPHAS[link].extend(alpha_parameters[i])
            BETAS[link].extend(beta_parameters[i])
    return ALPHAS, BETAS

mouse_type_input = input('What mouse type would you like? (16p_rev, 16p_var, WT_rev, WT_var)\n')
mouse_type_dict = {
    '16p_rev': links_16p_rev,
    '16p_var': links_16p_var,
    'WT_rev': links_WT_rev,
    'WT_var': links_WT_var
}

selected_mouse_type = mouse_type_dict.get(mouse_type_input, [])

if not selected_mouse_type:
    print("Invalid mouse type selected.")
else:
    A, B = alphasbetas(selected_mouse_type)

    for mouse_link in selected_mouse_type:
        # Plot alpha parameters
        phases = list(A[mouse_link].keys())
        values = [A[mouse_link][phase][0] for phase in phases]  # Assuming each phase has one value
        plt.plot(phases, values, label=f'Alpha: {mouse_link}')
    
    plt.legend()
    plt.title('Alpha Parameters for ' + mouse_type_input)
    plt.xlabel('Phases')
    plt.ylabel('Alpha Value')
    plt.show()

    for mouse_link in selected_mouse_type:
        # Plot beta parameters
        phases = list(B[mouse_link].keys())
        values = [B[mouse_link][phase][0] for phase in phases]  # Assuming each phase has one value
        plt.plot(phases, values, label=f'Beta: {mouse_link}')
    
    plt.legend()
    plt.title('Beta Parameters for ' + mouse_type_input)
    plt.xlabel('Phases')
    plt.ylabel('Beta Value')
    plt.show()
