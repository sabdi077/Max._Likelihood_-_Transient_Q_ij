import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, Num_trials):
        self.states = []
        self.actions = []
        self.R = []
        self.Num_trials = Num_trials
        self.Q = {'6kHz': {'L': [0], 'R': [0], 'N': [0]}, '10kHz': {'L': [0], 'R': [0], 'N': [0]}}
    
    def Generate_SAR(self, A, B):
        rule_reversed = False
        Q_history = []
        trial_counter=0
        recent_CORRECT=[]
        for t in range(self.Num_trials):
            state = '6kHz' if np.random.rand() < 0.5 else '10kHz'
            self.states.append(state)
            
            P_L = np.exp(B * self.Q[state]['L'][t]) / (np.exp(B * self.Q[state]['L'][t]) + np.exp(B * self.Q[state]['R'][t]) + np.exp(B * self.Q[state]['N'][t]))
            P_R = np.exp(B * self.Q[state]['R'][t]) / (np.exp(B * self.Q[state]['L'][t]) + np.exp(B * self.Q[state]['R'][t]) + np.exp(B * self.Q[state]['N'][t]))
            P_N = 1 - (P_L + P_R)
            
            r1 = np.random.rand()
            if r1 < P_L:
                action = 'L'
            elif r1 < (P_L + P_R):
                action = 'R'
            else:
                action = 'N'
            
            self.actions.append(action)
            
            if not rule_reversed:
                correct = (state == '6kHz' and action == 'L') or (state == '10kHz' and action == 'R')
            else:
                correct = (state == '6kHz' and action == 'R') or (state == '10kHz' and action == 'L')
            
            r2 = np.random.rand()
            if correct:
                reward = 1 if r2 < 0.9 else 0
            else:
                reward = 1 if r2 > 0.9 else 0
            
            self.R.append(reward)

            self.Q_Algorithm(A, t, state, action, reward)
            Q_history.append(self.Q.copy())
            
            '''if len(recent_CORRECT)>20:
                recent_CORRECT.pop(0)
            
            if sum(recent_CORRECT)>=19 and trial_counter==0:
                trial_counter = 250
            
            if trial_counter>0:
                trial_counter -= 1
                if trial_counter == 0:
                    rule_reversed = not rule_reversed
                    recent_CORRECT=[]'''
            
        return self.states, self.actions, self.R, Q_history
    
    def Q_Algorithm(self, A, k, state, action, reward):
        if state == '6kHz':
            other_state = '10kHz'
        else:
            other_state = '6kHz'
        
        other_actions = [a for a in ['L', 'R', 'N'] if a != action]

        self.Q[state][action].append(self.Q[state][action][k] + A * (reward - self.Q[state][action][k]))
        for other_action in other_actions:
            self.Q[state][other_action].append(self.Q[state][other_action][k])
            self.Q[other_state][other_action].append(self.Q[other_state][other_action][k])


    def log_likelihood(self, A, B, states, actions, rewards):  
        log_likelihood = []
        Q = {'6kHz': {'L': [0], 'R': [0], 'N': [0]}, '10kHz': {'L': [0], 'R': [0], 'N': [0]}}

        for t in range(self.Num_trials):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            P_L = np.exp(B * Q[state]['L'][t]) / (np.exp(B * Q[state]['L'][t]) + np.exp(B * Q[state]['R'][t]) + np.exp(B * Q[state]['N'][t]))
            P_R = np.exp(B * Q[state]['R'][t]) / (np.exp(B * Q[state]['L'][t]) + np.exp(B * Q[state]['R'][t]) + np.exp(B * Q[state]['N'][t]))
            P_N = 1 - (P_L + P_R)
            
            if action == 'L':
                P_t = P_L
            elif action == 'R':
                P_t = P_R
            else:
                P_t = P_N
            
            if P_t <= 1e-15:
                log_likelihood.append(-1e15)
            else:
                log_likelihood.append(np.log(P_t))

            if state == '6kHz':
                other_state = '10kHz'
            else:
                other_state = '6kHz'
            
            other_actions = [a for a in ['L', 'R', 'N'] if a != action]

            Q[state][action].append(Q[state][action][t] + A * (reward - Q[state][action][t]))
            for other_action in other_actions:
                Q[state][other_action].append(Q[state][other_action][t])
                Q[other_state][other_action].append(Q[other_state][other_action][t])
        
        return log_likelihood

    def neg_log_likelihood(self, params, states, actions, rewards):
        A, B = params
        log_likelihood = self.log_likelihood(A, B, states, actions, rewards)
        return -np.sum(log_likelihood)

class Runs:
    def __init__(self, Num_trials, num_est):
        self.Num_trials = Num_trials
        self.num_est = num_est
        self.alpha_estimates_list = []
        self.beta_estimates_list = []

    def estimate_alphas_betas(self, random_alphas, random_betas):
        for j in range(len(random_alphas)):
            alpha_estimates_j = []
            beta_estimates_j = []
            sim = Simulation(self.Num_trials)
            states, actions, rewards, _ = sim.Generate_SAR(random_alphas[j], random_betas[j])
            
            for i in range(self.num_est):
                initial_guess = [np.random.uniform(0, 1), np.random.uniform(0, 10)]
                bounds = [(0, 2), (0, 34)]
                
                result = minimize(lambda x: sim.neg_log_likelihood(x, states, actions, rewards), initial_guess, bounds=bounds)
                alpha_fit, beta_fit = result.x
                alpha_estimates_j.append(alpha_fit)
                beta_estimates_j.append(beta_fit)
            
            self.alpha_estimates_list.append(np.mean(alpha_estimates_j))
            self.beta_estimates_list.append(np.mean(beta_estimates_j))
        
        return self.alpha_estimates_list, self.beta_estimates_list
    
    def plot_Q_values(self, Q_history):
        times = range(len(Q_history))
        
        for state in ['6kHz', '10kHz']:
            for action in ['L', 'R', 'N']:
                Q_values = [Q[state][action][-1] for Q in Q_history]
                plt.plot(times, Q_values, label=f'{state} - {action}')
        
        plt.xlabel('Time')
        plt.ylabel('Q-value')
        plt.legend()
        plt.title('Q-values over time')
        plt.show()

# Running the simulation
r = Runs(3000, 5)
'''alphas = np.linspace(0.1, 0.99, 10).tolist()
betas = np.linspace(0.1, 10, 10).tolist()
alpha_ests, beta_ests = r.estimate_alphas_betas(alphas, betas)

plt.plot(alphas, alpha_ests, label='Estimated Alphas')
plt.xlabel('True Alphas')
plt.ylabel('Estimated Alphas')
plt.legend()
plt.show()

plt.plot(betas, beta_ests, label='Estimated Betas')
plt.xlabel('True Betas')
plt.ylabel('Estimated Betas')
plt.legend()
plt.show()'''

# Plotting Q-values over time
sim = Simulation(3000)
states, actions, rewards, Q_history = sim.Generate_SAR(0.5, 5.0)
r.plot_Q_values(Q_history)
