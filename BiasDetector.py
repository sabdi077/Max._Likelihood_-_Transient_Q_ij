import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

def is_bias_condition(S, A, t, tau, A_bias):
    I_6 = sum((S[i] == '6kHz') and (A[i] == A_bias) for i in range(t, t + tau))
    I_10 = sum((S[i] == '10kHz') and (A[i] == A_bias) for i in range(t, t + tau))
    return (abs(I_6 - I_10) / tau < 0.3) and (I_6 + I_10 == tau)

def detector(link, tau, A_bias):
    
    df = pd.read_csv('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/' + link)
    S = df['tone_freq'].replace({6000: '6kHz', 10000: '10kHz'}).reset_index(drop=True)
    A = df['response'].reset_index(drop=True)
    n = len(A)
    Bias_intervals = [False] * n
    NonBias_intervals = [False] * n

    t = 0
    while t <= n - tau:
        if is_bias_condition(S, A, t, tau, A_bias):
            k = 0
            while t + tau + k < n and is_bias_condition(S, A, t + k, tau, A_bias):
                k += 1
            end = t + tau + k - 1
            for i in range(t, end + 1):
                Bias_intervals[i] = True
            t = end + 1
        else:
            k = 0
            while t + tau + k < n and not is_bias_condition(S, A, t + k, tau, A_bias):
                k += 1
            end = t + tau + k - 1
            for i in range(t, end + 1):
                NonBias_intervals[i] = True
            t = end + 1
    
    return np.sum(Bias_intervals)

def process_mouse_type(args):
    mouse_type, ids, tau, biasA = args
    bias_counts = []
    for ID in ids:
        link = f'mouse_data_{ID}_{mouse_type.lower().replace(" ", "_")}_prob.csv'
        bias_counts.append(detector(link, tau, biasA))
    return mouse_type, tau, np.mean(bias_counts)

if __name__ == "__main__":
   
    df = pd.read_excel('/Users/saadabdisalam/Documents/MouseData_and_Analysis2024-2025/Mouse_DATA.xlsx')
    sixteenP_rev = df['16p_rev'].iloc[:5].dropna().astype(int).tolist()
    sixteenP_var = df['16p_var'].iloc[:5].dropna().astype(int).tolist()
    WT_rev = df['WT_rev'].iloc[:5].dropna().astype(int).tolist()
    WT_var = df['WT_var'].iloc[:5].dropna().astype(int).tolist()

   
    taus = range(4, 50)  
    mouse_types = ['16p11.2 REV', '16p11.2 VAR', 'WT REV', 'WT VAR']
    mouse_ids = [sixteenP_rev, sixteenP_var, WT_rev, WT_var]

    biasA = input("What do you want your action bias to be\n")

    
    tasks = [(mouse_type, ids, tau, biasA) for mouse_type, ids in zip(mouse_types, mouse_ids) for tau in taus]

   
    results = {mouse_type: [] for mouse_type in mouse_types}


    with Pool() as pool:
        for mouse_type, tau, mean_bias_count in pool.map(process_mouse_type, tasks):
            results[mouse_type].append((tau, mean_bias_count))

    
    plt.figure(figsize=(14, 8))

    for mouse_type in mouse_types:
        sorted_results = sorted(results[mouse_type])
        taus, means = zip(*sorted_results)
        plt.plot(taus, means, label=mouse_type)

    plt.xlabel('Tau')
    plt.ylabel('Mean Bias Count')
    plt.title('Mean Bias Count for Different Types of Mice and Tau Values')
    plt.legend()
    plt.show()


