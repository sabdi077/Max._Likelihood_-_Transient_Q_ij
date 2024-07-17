import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def is_bias_condition(S, A, t, tau, A_bias):
    I_6 = sum((S[i] == '6kHz') and (A[i] == A_bias) for i in range(t, t + tau))
    I_10 = sum((S[i] == '10kHz') and (A[i] == A_bias) for i in range(t, t + tau))
    return (abs(I_6 - I_10) / tau < 0.3) and (I_6 + I_10 == tau)

def Detector(link, tau, A_bias):
    df = pd.read_csv('/Users/saadabdisalam/Documents/MouseDataAnalysis2023-2024/' + link)
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
    
    return Bias_intervals, NonBias_intervals



def add_bias_columns(link, L_bias_intervals, N_bias_intervals, R_bias_intervals):
    df = pd.read_csv('/Users/saadabdisalam/Documents/MouseDataAnalysis2023-2024/' + link)
    df['L_Bias_Intervals'] = L_bias_intervals
    df['N_Bias_Intervals'] = N_bias_intervals
    df['R_Bias_Intervals'] = R_bias_intervals
    df.to_csv('/Users/saadabdisalam/Documents/MouseDataAnalysis2023-2024/' + link, index=False)

# Load the main data file
df = pd.read_excel('/Users/saadabdisalam/Documents/MouseDataAnalysis2023-2024/Mouse_DATA.xlsx')
sixteenP_rev = df['16p_rev'].dropna().astype(int).tolist()
sixteenP_var = df['16p_var'].dropna().astype(int).tolist()
WT_rev = df['WT_rev'].dropna().astype(int).tolist()
WT_var = df['WT_var'].dropna().astype(int).tolist()

s = int(input("What do you want your tau to be?\n"))

# Process each file and add bias intervals columns
for ID in sixteenP_rev:
    link = f'mouse_data_{ID}_16p11.2_rev_prob.csv'
    L_bias_trials, _ = Detector(link, s, 'L')
    N_bias_trials, _ = Detector(link, s, 'N')
    R_bias_trials, _ = Detector(link, s, 'R')
    add_bias_columns(link, L_bias_trials, N_bias_trials, R_bias_trials)

for ID in sixteenP_var:
    link = f'mouse_data_{ID}_16p11.2_var_prob.csv'
    L_bias_trials, _ = Detector(link, s, 'L')
    N_bias_trials, _ = Detector(link, s, 'N')
    R_bias_trials, _ = Detector(link, s, 'R')
    add_bias_columns(link, L_bias_trials, N_bias_trials, R_bias_trials)

for ID in WT_rev:
    link = f'mouse_data_{ID}_WT_rev_prob.csv'
    L_bias_trials, _ = Detector(link, s, 'L')
    N_bias_trials, _ = Detector(link, s, 'N')
    R_bias_trials, _ = Detector(link, s, 'R')
    add_bias_columns(link, L_bias_trials, N_bias_trials, R_bias_trials)

for ID in WT_var:
    link = f'mouse_data_{ID}_WT_var_prob.csv'
    L_bias_trials, _ = Detector(link, s, 'L')
    N_bias_trials, _ = Detector(link, s, 'N')
    R_bias_trials, _ = Detector(link, s, 'R')
    add_bias_columns(link, L_bias_trials, N_bias_trials, R_bias_trials)

print("Bias intervals added to all respective CSV files.")