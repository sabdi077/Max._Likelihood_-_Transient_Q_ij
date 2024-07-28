import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
class Simulation:
    def __init__(self, Num_trials, p):
        self.states = []
        self.actions = []
        self.R = []
        self.Num_trials = Num_trials
        self.Q = {'6kHz': {'L': [], 'R': [], 'N': []}, '10kHz': {'L': [], 'R': [], 'N': []}}
        self.p=p
    def Generate_SAR(self, A, B):
        rule_reversed = False
        Q_history = []
        trial_counter = 0
        recent_CORRECT = []
        CORRECT=0
        Expert = []
        # Initialize Q-values for all actions and states
        for state in self.Q:
            for action in self.Q[state]:
                self.Q[state][action] = [0]  # Start with a single initial Q-value

        for t in range(self.Num_trials):
            state = '6kHz' if np.random.rand() < 0.5 else '10kHz'
            self.states.append(state)

            P_L = np.exp(B * self.Q[state]['L'][-1]) / (np.exp(B * self.Q[state]['L'][-1]) + np.exp(B * self.Q[state]['R'][-1]) + np.exp(B * self.Q[state]['N'][-1]))
            P_R = np.exp(B * self.Q[state]['R'][-1]) / (np.exp(B * self.Q[state]['L'][-1]) + np.exp(B * self.Q[state]['R'][-1]) + np.exp(B * self.Q[state]['N'][-1]))
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
            if (correct and (r2<self.p)) or (not correct and (r2>self.p)):
                reward = 1
            else:
                reward=0
            self.R.append(reward)
            recent_CORRECT.append(correct)
            CORRECT+=correct
            self.Q_Algorithm(A, t, state, action, reward)
              
            if len(recent_CORRECT)>20:
                recent_CORRECT.pop(0)
            
            if sum(recent_CORRECT)>=19 and trial_counter==0:
                trial_counter = 250
                expert_start = t
            
            if trial_counter>0:
                trial_counter -= 1
                if trial_counter == 0:
                    expert_end = t
                    rule_reversed = not rule_reversed
                    recent_CORRECT=[]
                    Expert.append([expert_start, expert_end])
            
            Q_history.append({s: {a: self.Q[s][a][-1] for a in self.Q[s]} for s in self.Q})

        return self.states, self.actions, self.R, Q_history, CORRECT/self.Num_trials, Expert_series
    
    def Q_Algorithm(self, A, t, state, action, reward):
        current_Q = self.Q[state][action][-1]  # Get the last Q-value
        new_Q = current_Q + A * (reward - current_Q)
        self.Q[state][action].append(new_Q)

        # For actions not taken, carry forward the previous Q-value
        for a in ['L', 'R', 'N']:
            if a != action:
                self.Q[state][a].append(self.Q[state][a][-1])

    def plot_Q_values(self, Q_history):
        times = range(len(Q_history))
        
        for state in ['6kHz', '10kHz']:
            for action in ['L', 'R', 'N']:
                Q_values = [Q[state][action] for Q in Q_history]
                plt.plot(times, Q_values, label=f'{state} - {action}')
        
        plt.xlabel('Time')
        plt.ylabel('Q-value')
        plt.legend()
        plt.title('Q-values over time')
        plt.show()

    def log_likelihood(self, A, B, states, actions, rewards):
        log_likelihood = []
        # Initialize the Q matrix for all actions in both states
        Q = {'6kHz': {'L': [0]*self.Num_trials, 'R': [0]*self.Num_trials, 'N': [0]*self.Num_trials},
            '10kHz': {'L': [0]*self.Num_trials, 'R': [0]*self.Num_trials, 'N': [0]*self.Num_trials}}

        for t in range(self.Num_trials):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            # Compute the total exponential sum for normalization
            total_exp = sum(np.exp(B * Q[state][a][t]) for a in ['L', 'R', 'N'])
            P_L = np.exp(B * Q[state]['L'][t]) / total_exp
            P_R = np.exp(B * Q[state]['R'][t]) / total_exp
            P_N = 1 - (P_L + P_R)

            # Select the probability associated with the action taken
            P_t = {'L': P_L, 'R': P_R, 'N': P_N}[action]
            log_likelihood.append(np.log(P_t) if P_t > 1e-15 else -1e15)

            # Update Q values for next time step
            if t < self.Num_trials - 1:
                for a in ['L', 'R', 'N']:
                    if a == action:
                        Q[state][a][t+1] = Q[state][a][t] + A * (reward - Q[state][a][t])
                    else:
                        Q[state][a][t+1] = Q[state][a][t]

        return log_likelihood


    def neg_log_likelihood(self, params, states, actions, rewards):
        A, B = params
        log_likelihood = self.log_likelihood(A, B, states, actions, rewards)
        return -np.sum(log_likelihood)

class Runs:
    def __init__(self, Num_trials, num_est, p):
        self.Num_trials = Num_trials
        self.num_est = num_est
        self.alpha_estimates_list = []
        self.beta_estimates_list = []
        self.p=p
    def estimate_alphas_betas(self, random_alphas, random_betas):
        for j in range(len(random_alphas)):
            alpha_estimates_j = []
            beta_estimates_j = []
            sim = Simulation(self.Num_trials, self.p)
            states, actions, rewards, _,_ = sim.Generate_SAR(random_alphas[j], random_betas[j])
            
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
# Q change over time plots
'''r = Runs(3000, 5, 0.7)
alphas = np.linspace(0.1, 0.99, 10).tolist()
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
# the N(correct)/num_trials and N(R=1)/num_trials vs. P(R=1|(ca, cs))
'''sim = Simulation(1000, 0.7)
states, actions, rewards, Q_history, percent_correct = sim.Generate_SAR(0.6, 3.3)
sim.plot_Q_values(Q_history)'''

'''ps = np.linspace(0.5, 1, 1000)
L=[]
U=[]

for p in ps:
    sim = Simulation(1000, p)
    states, actions, rewards, Q_history, percent_correct = sim.Generate_SAR(0.9, 1.3)
    L.append(percent_correct)
    U.append(np.sum(rewards)/sim.Num_trials)
plt.scatter(ps, L, label='correct')
#plt.scatter(ps, U, label='reward')
plt.xlabel('P(R=1|(correct (s,a)))')
plt.ylabel('N(correct = 1) and N(reward = 1)')
plt.legend()
plt.show()'''


