import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

class Simulation:
    def __init__(self, start_trial, end_trial, Q_i, df):
        
        self.end_trial = end_trial
        self.start_trial = start_trial
        self.Q_i = Q_i
        self.R = df['rew_t'].iloc[start_trial:end_trial].reset_index(drop=True).astype(int)
        self.states = df['tone_freq'].iloc[start_trial:end_trial].replace({6000: '6kHz', 10000: '10kHz'}).reset_index(drop=True)
        self.actions = df['response'].iloc[start_trial:end_trial].reset_index(drop=True)
        self.correct_act = df['corr_choice'].iloc[start_trial:end_trial].reset_index(drop=True)
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
        Q_history = [self.Q_i.copy()]

        for t in range(self.end_trial - self.start_trial):
            q_current = Q_history[t][self.states[t]][self.actions[t]]
            opposite_actions = {'L': ['R', 'N'], 'R': ['N', 'L'], 'N': ['L', 'R']}
            q_opposite1 = Q_history[t][self.states[t]][opposite_actions[self.actions[t]][0]]
            q_opposite2 = Q_history[t][self.states[t]][opposite_actions[self.actions[t]][1]]
            q_diff1 = q_opposite1 - q_current
            q_diff2 = q_opposite2 - q_current
            P_t = 1 / (1 + np.exp(B * q_diff1) + np.exp(B * q_diff2))
            log_likelihood.append(np.log(P_t))

            reward = self.R[t]
            Q_history.append(Q_history[t].copy())
            Q_history[t+1][self.states[t]][self.actions[t]] += A * (reward - Q_history[t][self.states[t]][self.actions[t]])

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
        self.A_range = np.linspace(10**(-50), 0.99, 200)
        self.B_range = np.linspace(1, 30, 200)

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

def process_trial(args):
    start_trial, Q_i, df, link = args
    end_trial = start_trial + 307
    sim2 = Simulation(start_trial, start_trial + 20, Q_i, df)
    mle = MLE(start_trial, end_trial, Q_i, df)
    A_optimal, B_optimal, ll, P = mle.func()
    Q_i = sim2.Delta(A_optimal)
    return 100*A_optimal, B_optimal, Q_i

def generate_switch_list(state_list):
    switch_list = []
    for i in range(1, len(state_list)):
        if state_list[i] != state_list[i-1]:
            switch_list.append(i)
    return switch_list

def gen_optimized(ws, ft, tR, switch_list):
    final = (ft - tR) // ws   
    COL = ["blue"] * (final + 1)   
    
    for k in range(final + 1):   
        start = k * ws
        end = start + tR
        for m in switch_list:
            if start <= m < end:   
                COL[k] = "red"   
                break   
                
    return COL

if __name__ == "__main__":


    link = input("Write your link to a csv file\n")
    print(f"{os.getcwd()}/{link}")
    df = pd.read_csv('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/'+ link)
    RewSeq = df['rew_t'].iloc[:].reset_index(drop=True).astype(int)

    Q_i = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
    x = range(0, (len(RewSeq) - 307) - 1, 20)

    
    expert_list = df['rule'].iloc[:].reset_index(drop=True).tolist()
    switch_list = generate_switch_list(expert_list)
    
    
    COL = gen_optimized(20, len(RewSeq), 307, switch_list)

    args = [(i, Q_i, df, link) for i in x]

    with Pool() as pool:
        results = pool.map(process_trial, args)

    A, B, Q_i = zip(*results)

     
    plt.figure(figsize=(10, 4))
    plt.xlabel('Trials')
    plt.ylabel(r'$\alpha$ (Optimal A)')
    plt.scatter(x, A, c=COL[:len(A)], label='Optimal A over Trials')  
    plt.legend()
    plt.show()

     
    plt.figure(figsize=(10, 4))
    plt.xlabel('Trials')
    plt.ylabel(r'$\beta$ (Optimal B)')
    plt.scatter(x, B, c=COL[:len(B)], label='Optimal B over Trials')  
    plt.legend()
    plt.show()
