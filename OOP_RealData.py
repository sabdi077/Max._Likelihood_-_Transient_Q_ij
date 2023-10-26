import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('expert_mice_all_trials.csv')

class Simulation:
    def __init__(self, start_trial, end_trial, Q_i):
        self.end_trial = end_trial
        self.start_trial = start_trial
        self.Q_i = Q_i
        self.R = df['rew_t'].iloc[start_trial:end_trial].reset_index(drop=True).astype(int)
        self.states = df['tone_freq'].iloc[start_trial:end_trial].reset_index(drop=True)
        for t in range(end_trial - start_trial):
            self.states[t]='6kHz' if self.states[t]==6000 else '10kHz'
        self.actions = df['response'].iloc[start_trial:end_trial].reset_index(drop=True)

    def Delta(self, A):
        N = len(self.states)

        # Define the initial Q values as a dictionary
        Q = self.Q_i # intialiaztion
        Q_history = [Q.copy()]  # Keep a history of Q at each time step

        # Iterate over the trials
        for t in range(N):
            # Get the current state and action
            state, action = self.states[t], self.actions[t]

            # Update the corresponding element of the matrix
            Q[state][action] = Q[state][action] + A * (self.R[t] - Q[state][action])

            # Append the updated Q to the history
            Q_history.append(Q.copy())

        # Extract the Q values for each state-action pair for the final trial
        final_trial_Q = Q  # This is the Q values for the final trial

        return final_trial_Q

    def log(self, A, B):  
        log_likelihood = []
        #if self.start_trial==0:
        Q_history = [self.Q_i]
        #else:
            #A_i,_,_=MLE(0,self.start_trial).func()
            #Q_history=self.delta_learning_rule(A_i, self.start_trial)

        for t in range(self.end_trial - self.start_trial):
            q_current = Q_history[t][self.states[t]][self.actions[t]]
            if self.actions[t] == 'L': 
                opposite_action1='R'
                opposite_action2='N'
            elif self.actions[t] == 'R': 
                opposite_action1='N'
                opposite_action2='L'
            else:
                opposite_action1='L'
                opposite_action2='R'
            q_opposite1 = Q_history[t][self.states[t]][opposite_action1]
            q_opposite2 = Q_history[t][self.states[t]][opposite_action2]
            q_diff1 = q_opposite1 - q_current
            q_diff2 = q_opposite2 - q_current
            P_t = 1 / (1 + np.exp(B * q_diff1)+np.exp(B * q_diff2))
            if P_t <= 1e-15:  # Avoiding too small values
                log_likelihood.append(-1e15)  # log(small num)= neg large num
            else:
                log_likelihood.append(np.log(P_t))

            # Updating Q-values
            reward = self.R[t]
            Q_history.append(Q_history[t].copy())
            Q_history[t+1][self.states[t]][self.actions[t]] += A * (reward - Q_history[t][self.states[t]][self.actions[t]])

        return log_likelihood

    def neg_log_likelihood(self, params):
        A, B = params
        log_likelihood = self.log(A, B)
        return -np.sum(log_likelihood)

class MLE:

    def __init__(self, start_trial, end_trial, Q_i):
        self.start_trial=start_trial
        self.end_trial=end_trial
        self.Q_i = Q_i
        self.A_range = np.linspace(0, 1, 100)
        self.B_range = np.linspace(0, 10, 100)

    def func(self):
        real=Simulation(self.start_trial, self.end_trial, self.Q_i)
        
        # Initialize neg_log_likelihoods as 2D array of zeros
        log_likelihoods = np.zeros((len(self.A_range), len(self.B_range)))
        # Fill in neg_log_likelihoods
        for i, A in enumerate(self.A_range):
            for j, B in enumerate(self.B_range):
                log_likelihoods[i, j] = -real.neg_log_likelihood([A, B])
        ll= np.exp(log_likelihoods - np.max(log_likelihoods))
        i_max, j_max = np.unravel_index(np.argmax(ll), log_likelihoods.shape)

        return self.A_range[i_max], self.B_range[j_max], ll
    
    #def Struct(self):

 