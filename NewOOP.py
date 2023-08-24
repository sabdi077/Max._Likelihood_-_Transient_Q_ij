import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, Num_trials):
        self.states = []
        self.actions = []
        self.R = []
        self.Num_trials = Num_trials

    def generate_states_actions_rewards(self, A, B):
        Q = {'6kHz': {'L': 0, 'R': 0}, '10kHz': {'L': 0, 'R': 0}}
        Q_history = [Q.copy()]

        for t in range(self.Num_trials):
            random_val = np.random.uniform(0, 1)
            random_val1 = np.random.uniform(0, 1)
            random_val2 = np.random.uniform(0, 1)

            state = '6kHz' if random_val < 0.5 else '10kHz'
            self.states.append(state)

            P_L = np.exp(B * Q[state]['L']) / (np.exp(B * Q[state]['L']) + np.exp(B * Q[state]['R']))
            action = 'L' if random_val1 <= P_L else 'R'
            self.actions.append(action)

            if ((state == '6kHz' and action == 'L') or (state == '10kHz' and action == 'R')) and (random_val2 < 0.9):
                reward = 1
            elif ((state == '6kHz' and action == 'R') or (state == '10kHz' and action == 'L')) and (random_val2 > 0.9):
                reward = 1
            else:
                reward = 0

            self.R.append(reward)
            Q[state][action] = Q[state][action] + A * (reward - Q[state][action])
            Q_history.append(Q.copy())
        return self.states, self.actions, self.R, Q_history

    def delta_learning_rule(self, A):
        Q = {'6kHz': {'L': 0, 'R': 0}, '10kHz': {'L': 0, 'R': 0}}
        for t in range(self.Num_trials):
            Q[self.states[t]][self.actions[t]] = Q[self.states[t]][self.actions[t]] + A * (
                        self.R[t] - Q[self.states[t]][self.actions[t]])
        return Q

    def log(self, A, B, states, actions):  
        log_likelihood = []
        Q_history = [{'6kHz': {'L': 0, 'R': 0}, '10kHz': {'L': 0, 'R': 0}}]

        for t in range(self.Num_trials):
            q_current = Q_history[t][states[t]][actions[t]]
            opposite_action = 'R' if actions[t] == 'L' else 'L'
            q_opposite = Q_history[t][states[t]][opposite_action]
            q_diff = q_opposite - q_current

            P_t = 1 / (1 + np.exp(B * q_diff))
            if P_t <= 1e-15:  # Avoiding too small values
                log_likelihood.append(-1e15)  # log(small num)= neg large num
            else:
                log_likelihood.append(np.log(P_t))

            # Updating Q-values
            reward = self.R[t]
            Q_history.append(Q_history[t].copy())
            Q_history[t+1][states[t]][actions[t]] += A * (reward - Q_history[t][states[t]][actions[t]])

        return log_likelihood

    def neg_log_likelihood(self, params, states, actions):
        A, B = params
        log_likelihood = self.log(A, B, states, actions)
        return -np.sum(log_likelihood)


class Runs:
    def __init__(self, Num_trials, Num_mice, num_est):
        self.Num_trials = Num_trials
        self.num_est = num_est
        self.Num_mice = Num_mice
        self.alpha_estimates_list = []
        self.beta_estimates_list = []

    def estimate_alphas_betas(self, random_alphas, random_betas):
        for j in range(self.Num_mice):
            alpha_estimates_j = []
            beta_estimates_j = []

            for i in range(self.num_est):
                sim = Simulation(self.Num_trials)
                
                # Generating synthetic data based on given alphas and betas
                states, actions, _, _ = sim.generate_states_actions_rewards(random_alphas[j], random_betas[j])

                # Providing some initial guess for alpha and beta
                initial_guess = [np.random.uniform(0.1,1.5), np.random.uniform(0.2,10.1)]

                # Defining the bounds for alpha and beta
                bounds = [(0.1, 1.5), (0.2, 10.1)]

                # Using minimize function to estimate alpha and beta for this synthetic data
                result = minimize(lambda x: sim.neg_log_likelihood(x, states, actions), initial_guess, bounds=bounds)

                # Appending the estimates to the respective lists for this specific random_alphas[j]
                alpha_fit, beta_fit = result.x
                alpha_estimates_j.append(alpha_fit)
                beta_estimates_j.append(beta_fit)

            # Appending the alpha and beta estimates lists for this specific random_alphas[j] to the main lists
            self.alpha_estimates_list.append(alpha_estimates_j)
            self.beta_estimates_list.append(beta_estimates_j)

        return self.alpha_estimates_list, self.beta_estimates_list
