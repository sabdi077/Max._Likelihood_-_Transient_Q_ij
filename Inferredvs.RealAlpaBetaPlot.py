from OOP import RunsOverSimulation
import numpy as np
import matplotlib.pyplot as plt

def generate_random_alpha_beta(N):
    alphas = np.random.uniform(0, 1, N)
    betas = np.random.uniform(0, 9, N)
    return alphas, betas

N = 1000  # Seting the number of randomly generated alpha's and beta's
num_runs = 50
e = 0.2

# Generate N random alpha and beta values
real_alphas, real_betas = generate_random_alpha_beta(N)

# Create an instance of RunsOverSimulation
runs_simulation = RunsOverSimulation(N, num_runs)

# Estimate alphas and betas for each randomly generated alpha and beta
alpha_estimates_list, beta_estimates_list = runs_simulation.estimate_alphas_betas(real_alphas, real_betas, e)

# Plot alpha_estimates vs. random_alphas
plt.figure(figsize=(8, 6))
for j in range(N):
    plt.plot(num_runs*[real_alphas[j]], alpha_estimates_list[j], 'bo', alpha=0.5)

plt.xlabel('Randomly Generated Alphas')
plt.ylabel('Alpha Estimates')
plt.title('Alpha Estimates for Randomly Generated Alphas')
plt.grid(True)
plt.show()

# Plot beta_estimates vs. random_betas
plt.figure(figsize=(8, 6))
for j in range(N):
    plt.plot(num_runs*[real_betas[j]], beta_estimates_list[j], 'ro', alpha=0.5)

plt.xlabel('Randomly Generated Betas')
plt.ylabel('Beta Estimates')
plt.title('Beta Estimates for Randomly Generated Betas')
plt.grid(True)
plt.show()

 