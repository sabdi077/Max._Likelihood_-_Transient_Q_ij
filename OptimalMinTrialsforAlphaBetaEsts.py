import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from multiprocessing import Pool

class Simulation:
    def __init__(self, Num_trials, p, count, Q_i):
        self.states = []
        self.actions = []
        self.R = []
        self.count = count
        self.Num_trials = Num_trials
        self.Q = Q_i.copy()  # Use copy to avoid side effects
        self.p = p

    def Generate_SAR(self, A, B):
        rule_reversed = False
        Q_history = []
        trial_counter = 0
        recent_CORRECT = []
        CORRECT = 0
        Expert = []

        for state in self.Q:
            for action in self.Q[state]:
                self.Q[state][action] = 0

        for t in range(self.Num_trials):
            state = '6kHz' if np.random.rand() < 0.5 else '10kHz'
            self.states.append(state)

            P_L = np.exp(B * self.Q[state]['L']) / (np.exp(B * self.Q[state]['L']) + np.exp(B * self.Q[state]['R']) + np.exp(B * self.Q[state]['N']))
            P_R = np.exp(B * self.Q[state]['R']) / (np.exp(B * self.Q[state]['L']) + np.exp(B * self.Q[state]['R']) + np.exp(B * self.Q[state]['N']))
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
            reward = 1 if (correct and (r2 < self.p)) or (not correct and (r2 > self.p)) else 0
            # recent_CORRECT.append(correct)
            self.R.append(reward)
            # CORRECT += correct
            self.Q_Algorithm(A, state, action, reward)

            '''if len(recent_CORRECT) > 19:
                recent_CORRECT.pop(0)

            if sum(recent_CORRECT) >= 19 and trial_counter == 0:
                trial_counter = self.count
                expert_start = t

            if trial_counter > 0:
                trial_counter -= 1
                if trial_counter == 0:
                    expert_end = t
                    rule_reversed = not rule_reversed
                    recent_CORRECT = []
                    Expert.append([expert_start, expert_end])'''

            Q_history.append({s: {a: self.Q[s][a] for a in self.Q[s]} for s in self.Q})

        return self.states, self.actions, self.R, Q_history, CORRECT / self.Num_trials, Expert

    def Q_Algorithm(self, A, state, action, reward):
        self.Q[state][action] += A * (reward - self.Q[state][action])

    def log_likelihood(self, A, B, states, actions, rewards):
        Q = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
        log_likelihood = []

        for t in range(self.Num_trials):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            q_current = Q[state][action]
            if action == 'L': 
                opposite_action1 = 'R'
                opposite_action2 = 'N'
            elif action == 'R': 
                opposite_action1 = 'N'
                opposite_action2 = 'L'
            else:
                opposite_action1 = 'L'
                opposite_action2 = 'R'
            
            q_opposite1 = Q[state][opposite_action1]
            q_opposite2 = Q[state][opposite_action2]

            logits = np.array([0, B * (q_opposite1 - q_current), B * (q_opposite2 - q_current)])
            P_t = np.exp(logits[0] - logsumexp(logits))

            log_likelihood.append(np.log(P_t) if P_t > 1e-15 else -1e15)

            if t < self.Num_trials - 1:
                for a in ['L', 'R', 'N']:
                    Q[state][a] += A * (reward - Q[state][a]) if a == action else 0

        return log_likelihood

    def neg_log_likelihood(self, params, states, actions, rewards):
        A, B = params
        log_likelihood = self.log_likelihood(A, B, states, actions, rewards)
        return -np.sum(log_likelihood)

class Runs:
    def __init__(self, Num_trials, num_est, p, count, Q_i):
        self.Num_trials = Num_trials
        self.num_est = num_est
        self.alpha_estimates_list = []
        self.beta_estimates_list = []
        self.p = p
        self.Q_i = Q_i
        self.count = count

    def estimate_alphas_betas(self, random_alphas, random_betas):
        alpha_estimates = []
        beta_estimates = []
        for j in range(len(random_alphas)):
            sim = Simulation(self.Num_trials, self.p, self.count, self.Q_i)
            states, actions, rewards, _, _, _ = sim.Generate_SAR(random_alphas[j], random_betas[j])
            alpha_estimates_j = []
            beta_estimates_j = []

            for _ in range(self.num_est):
                initial_guess = [np.random.uniform(0, 1), np.random.uniform(0, 10)]
                bounds = [(0, 1.5), (0, 11)]

                result = minimize(lambda x: sim.neg_log_likelihood(x, states, actions, rewards), initial_guess, bounds=bounds)
                alpha_fit, beta_fit = result.x
                alpha_estimates_j.append(alpha_fit)
                beta_estimates_j.append(beta_fit)

            alpha_estimates.append(np.mean(alpha_estimates_j))
            beta_estimates.append(np.mean(beta_estimates_j))

        return alpha_estimates, beta_estimates

def MSEvsTrial(args):
    real_alpha, real_beta, num_est, Q_i, max_trials = args
    MSE_ALPHA_list = []
    MSE_BETA_list = []

    for i in range(1, max_trials + 1):
        runs_simulation = Runs(i, num_est, 0.9, 250, Q_i)
        alpha_estimates, beta_estimates = runs_simulation.estimate_alphas_betas([real_alpha], [real_beta])

        alpha_squared_errors = (np.array(alpha_estimates) - real_alpha) ** 2
        beta_squared_errors = (np.array(beta_estimates) - real_beta) ** 2

        MSE_ALPHA = np.mean(alpha_squared_errors)
        MSE_BETA = np.mean(beta_squared_errors)

        MSE_ALPHA_list.append(MSE_ALPHA)
        MSE_BETA_list.append(MSE_BETA)

    return MSE_ALPHA_list, MSE_BETA_list

def find_min_trials(real_alphas, real_betas, num_mice, num_est, Q_i, threshold_alpha, threshold_beta, success_rate, max_trials):
    pool = Pool()
    args_list = [(real_alphas[j], real_betas[j], num_est, Q_i, max_trials) for j in range(num_mice)]

    MSE_matrices = pool.map(MSEvsTrial, args_list)
    
    pool.close()
    pool.join()

    MSE_ALPHA_matrix = np.array([m[0] for m in MSE_matrices])
    MSE_BETA_matrix = np.array([m[1] for m in MSE_matrices])

    num_trials = max_trials
    for t in range(1, max_trials + 1):
        alpha_condition = np.mean(MSE_ALPHA_matrix[:, t-1] <= threshold_alpha) >= success_rate
        beta_condition = np.mean(MSE_BETA_matrix[:, t-1] <= threshold_beta) >= success_rate
        if alpha_condition and beta_condition:
            num_trials = t
            break

    return num_trials, MSE_ALPHA_matrix, MSE_BETA_matrix

if __name__ == "__main__":
    num_mice = 10
    real_alphas = np.linspace(0, 1, num_mice)
    real_betas = np.linspace(0, 10, num_mice)

    threshold_alpha = 0.04  
    threshold_beta = 0.2
    success_rate = 1
    max_trials = 5000
    num_est = 10
    Q_i = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}

    num_trials, MSE_ALPHA_matrix, MSE_BETA_matrix = find_min_trials(real_alphas, real_betas, num_mice, num_est, Q_i, threshold_alpha, threshold_beta, success_rate, max_trials)

    Num_trials = np.arange(1, num_trials + 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(MSE_ALPHA_matrix[:, :num_trials], aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='MSE')
    plt.xlabel('Number of Trials')
    plt.ylabel(r'Real $\alpha$ index')
    plt.title(r'MSE of $\alpha$')
    plt.xticks(np.arange(0, num_trials, 10), np.arange(1, num_trials + 1, 10))
    plt.yticks(np.arange(0, num_mice, 1), np.round(real_alphas, 2))
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(MSE_BETA_matrix[:, :num_trials], aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='MSE')
    plt.xlabel('Number of Trials')
    plt.ylabel(r'Real $\beta$ index')
    plt.title(r'MSE of $\beta$')
    plt.xticks(np.arange(0, num_trials, 10), np.arange(1, num_trials + 1, 10))
    plt.yticks(np.arange(0, num_mice, 1), np.round(real_betas, 2))
    plt.show()

    print(f"The minimum number of trials needed for MSE(alpha) <= {threshold_alpha} for 100% of alphas: {num_trials}")
    print(f"The minimum number of trials needed for MSE(beta) <= {threshold_beta} for 100% of betas: {num_trials}")
