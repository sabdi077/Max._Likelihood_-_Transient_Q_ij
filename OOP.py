import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, N):
        self.states = []
        self.actions = []
        self.R = []
        self.N = N

    def generate_states_actions_rewards(self, B, A):
        Q = {'6kHz': {'L': 0, 'R': 0}, '10kHz': {'L': 0, 'R': 0}}
        Q_history = [Q.copy()]
        for t in range(self.N):
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
            elif ((state == '6kHz' and action == 'R') or (state == '10kHz' and action == 'L')) and (random_val2 < 1 - 0.9):
                reward = 1
            else:
                reward = 0

            self.R.append(reward)
            Q[state][action] = Q[state][action] + A * (reward - Q[state][action])
            Q_history.append(Q.copy())
        return self.states, self.actions, self.R, Q_history

    def delta_learning_rule(self, A):
        Q = {'6kHz': {'L': 0, 'R': 0}, '10kHz': {'L': 0, 'R': 0}}
        for t in range(self.N):
            Q[self.states[t]][self.actions[t]] = Q[self.states[t]][self.actions[t]] + A * (
                        self.R[t] - Q[self.states[t]][self.actions[t]])
        return Q

    def log(self, B, Q):  
        log = []
        epsilon = 1e-10  # small constant to avoid log(0)

        # Dictionary to map state-action pairs to Q-values
        q_map = {
            ('6kHz', 'L'): Q['6kHz']['L'],
            ('6kHz', 'R'): Q['6kHz']['R'],
            ('10kHz', 'L'): Q['10kHz']['L'],
            ('10kHz', 'R'): Q['10kHz']['R']
        }

        for t in range(self.N):
            current_q = q_map[(self.states[t], self.actions[t])]
            opposite_action = 'R' if self.actions[t] == 'L' else 'L'
            opposite_q = q_map[(self.states[t], opposite_action)]
            q = opposite_q - current_q

            P_t = 1 / (1 + np.exp(B * q + epsilon))

            log.append(np.log(P_t))

        return log

    def neg_log_likelihood(self, params, *args):
        A, B = params
        Q = self.delta_learning_rule(A)
        log_likelihood = self.log(B, Q)
        return -np.sum(log_likelihood)


class RunsOverSimulation:
    def __init__(self, N, num_runs):
        self.N = N
        self.num_runs = num_runs
        self.alpha_estimates_list = []
        self.beta_estimates_list = []

    def estimate_alphas_betas(self, random_alphas, random_betas, e):
        for j in range(self.N):
            alpha_estimates_j = []
            beta_estimates_j = []

            for i in range(self.num_runs):
                sim = Simulation(self.N)
                states, actions, R, _ = sim.generate_states_actions_rewards(random_betas[j], random_alphas[j])

                # Providing some initial guess for alpha and beta
                initial_guess = [random_alphas[j] + e, random_betas[j] - e]

                # Defining the bounds for alpha and beta
                alpha_bounds = (0, 1.5)
                beta_bounds = (0, 10.1)
                bounds = [alpha_bounds, beta_bounds]

                # Using minimize function to estimate alpha and beta for this synthetic data
                result = minimize(sim.neg_log_likelihood, initial_guess, args=(states, actions, R), bounds=bounds)

                # Appending the estimates to the respective lists for this specific random_alphas[j]
                alpha_fit = result.x[0]
                beta_fit = result.x[1]
                alpha_estimates_j.append(alpha_fit)
                beta_estimates_j.append(beta_fit)

            # Appending the alpha and beta estimates lists for this specific random_alphas[j] to the main lists
            self.alpha_estimates_list.append(alpha_estimates_j)
            self.beta_estimates_list.append(beta_estimates_j)

        return self.alpha_estimates_list, self.beta_estimates_list



