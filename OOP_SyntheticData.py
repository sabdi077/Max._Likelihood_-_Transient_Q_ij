import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from multiprocessing import Pool

class Simulation:
    def __init__(self, Num_trials):
        self.states = []
        self.actions = []
        self.R = []
        self.Num_trials = Num_trials

    def generate_states_actions_rewards(self, A, B):
        Q = {'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}
        Q_history = [Q.copy()]

        for t in range(self.Num_trials):
            r0 = np.random.uniform(0, 1)
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)

            state = '6kHz' if r0 < 0.5 else '10kHz'
            self.states.append(state)

            # Calculate probabilities
            P_L = np.exp(B * Q[state]['L']) / (np.exp(B * Q[state]['L']) + np.exp(B * Q[state]['R']) + np.exp(B * Q[state]['N']))
            P_R = np.exp(B * Q[state]['R']) / (np.exp(B * Q[state]['L']) + np.exp(B * Q[state]['R']) + np.exp(B * Q[state]['N']))
            P_N = 1 - (P_L + P_R)

            # Determine action based on cumulative probabilities
            if r1 < P_L:
                action = 'L'
            elif r1 < (P_L + P_R):
                action = 'R'
            else:
                action = 'N'

            self.actions.append(action)


            if ((state == '6kHz' and action == 'L') or (state == '10kHz' and action == 'R')) and (r2 < 0.9):
                reward = 1
            elif ((state == '6kHz' and action == 'R') or (state == '10kHz' and action == 'L') or (state == '10kHz' and action == 'N') or (state == '6kHz' and action == 'N')) and (r2 > 0.9):
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
        Q_history = [{'6kHz': {'L': 0, 'R': 0, 'N': 0}, '10kHz': {'L': 0, 'R': 0, 'N': 0}}]

        for t in range(self.Num_trials):
            q_current = Q_history[t][states[t]][actions[t]]
            if actions[t] == 'L': 
                opposite_action1='R'
                opposite_action2='N'
            elif actions[t] == 'R': 
                opposite_action1='N'
                opposite_action2='L'
            else:
                opposite_action1='L'
                opposite_action2='R'
            q_opposite1 = Q_history[t][states[t]][opposite_action1]
            q_opposite2 = Q_history[t][states[t]][opposite_action2]
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
                initial_guess = [np.random.uniform(0,1), np.random.uniform(0,10)]

                # Defining the bounds for alpha and beta
                bounds = [(0, 1), (0, 10)]

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
    


 
from OOP_SyntheticData import Runs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool

def MSEvsTrialForRange(args):
    i, real_alpha, real_beta, num_est = args
    runs_simulation = Runs(i, 1, num_est)  # One mouse, 100 estimates
    alpha_estimates, beta_estimates = runs_simulation.estimate_alphas_betas([real_alpha], [real_beta])

    Dev_MSE_alpha = [(alpha_estimates[0][j] - real_alpha) ** 2 for j in range(num_est)]
    Dev_MSE_beta = [(beta_estimates[0][j] - real_beta) ** 2 for j in range(num_est)]

    MSE_ALPHA = np.sum(Dev_MSE_alpha) / num_est
    MSE_BETA = np.sum(Dev_MSE_beta) / num_est

    return MSE_ALPHA, MSE_BETA

def fit_func(Num_trials, A, B):
    return np.exp(-(A * np.array(Num_trials)) + B)

def derivative_fit(Num_trials, A, B):
    return -A * np.exp(-(A * np.array(Num_trials)) + B)

def opt_trials(Num_trials, A, B):
    i = 0
    while i < len(Num_trials) and derivative_fit([Num_trials[i]], A, B) <= -0.01:
        i = i + 1
    if i < len(Num_trials):
        return Num_trials[i]
    else:
        return None

if __name__ == "__main__":
    num_est = 100
    real_alpha = 0.3
    real_beta = 3
    Num_trials = list(range(1, 501))
    pool = Pool(processes=4)  # Adjust the number of processes as needed

    args_list = [(i, real_alpha, real_beta, num_est) for i in range(1, 501)]
    results = pool.map(MSEvsTrialForRange, args_list)
    MSE_ALPHA_list, MSE_BETA_list = zip(*results)
    
    popt_alpha, pcov_alpha = curve_fit(fit_func, Num_trials, MSE_ALPHA_list, p0=[np.random.uniform(1, 10), np.random.uniform(1, 10)], maxfev=1000)
    popt_beta, pcov_beta = curve_fit(fit_func, Num_trials, MSE_BETA_list, p0=[np.random.uniform(1, 10), np.random.uniform(1, 10)], maxfev=1000)
    A_alpha = popt_alpha[0]
    B_alpha = popt_alpha[1]
    A_beta = popt_beta[0]
    B_beta = popt_beta[1]

    print(f"A_alpha={popt_alpha[0]}, B_alpha = {popt_alpha[1]}")
    print(f"A_beta={popt_beta[0]}, B_beta = {popt_beta[1]}")
    print(f"Smallest Trial for which MSE_alpha is min: {opt_trials(Num_trials, A_alpha, B_alpha)}")
    print(f"Smallest Trial for which MSE_beta is min: {opt_trials(Num_trials, A_beta, B_beta)}")
    plt.figure()
    plt.plot(Num_trials, MSE_ALPHA_list, label='MSE_ALPHA')
    plt.plot(Num_trials, fit_func(Num_trials, A_alpha, B_alpha))
    plt.xlabel('Number of Trials')
    plt.ylabel(r'MSE of $\alpha$')
    plt.title(r'MSE of $\alpha$ vs. Number of trials')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(Num_trials, MSE_BETA_list, label='MSE_BETA')
    plt.plot(Num_trials, fit_func(Num_trials, A_beta, B_beta))
    plt.xlabel('Number of Trials')
    plt.ylabel(r'MSE of $\beta$')
    plt.title(r'MSE of $\beta$ vs. Number of trials')
    plt.grid(True)
    plt.show()
