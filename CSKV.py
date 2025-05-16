import CSKVbasic as cob
import sympy as sp
import numpy as np
import time

def CSKV(param):
    start_time = time.time()

    # Unpack input parameters
    data_name = param[0]      # Dataset name
    epsilon = param[1]        # Privacy budget
    rate = param[2]           # Top-k rate
    gamma = param[3]          # Scaling parameter for value domain
    pad_ell = param[4]        # Padding length

    code_name = 'CSKV'

    # Define data paths
    k_path = f'./data/{data_name}/CSKV_{data_name}_k.txt'
    v_path = f'./data/{data_name}/CSKV_{data_name}_v.txt'
    true_path = f'./data/{data_name}/CSKV_{data_name}_id_count_avg_norm.txt'

    # Load user key-value pairs and ground truth
    data_k = cob.readtxt(k_path)
    data_v = cob.readtxt(v_path)
    true_count = cob.readtxt(true_path)  # Format: (key, count, average)

    n = len(data_k)         # Number of users
    d = len(true_count)     # Size of the key domain
    k = int(rate * d)       # Top-k candidate size

    # Convert input into key-value pair tuples per user
    kv = []
    for i in range(n):
        data_k[i] = list(map(int, data_k[i]))
        tmp = zip(data_k[i], data_v[i])
        kv.append(list(tmp))
    print("Assembling key-value pairs complete.")

    # Normalized labels: integers from 1 to d
    label = [i + 1 for i in range(d)]

    # Fixed parameters for discretization and perturbation
    delta = 10
    alpha = 10
    beta = 1

    # Padding key-value pairs per user
    PS = [cob.PPS(kv[i], d, label, pad_ell, delta) for i in range(n)]
    print("Padding complete.")

    # Privacy budget split: compute secondary epsilon e2
    e1 = epsilon
    e2 = sp.N((sp.log(
        (2 * d * sp.exp(epsilon * (d + 3) / 2) + sp.exp(2 * epsilon) - sp.exp(
            d * epsilon / 2) - 2 * sp.exp(epsilon * (d + 3) / 2)) / (
                2 * d * sp.exp(epsilon * (d + 1) / 2) - sp.exp(
            2 * epsilon) + sp.exp(
            d * epsilon / 2) - 2 * sp.exp(epsilon * (d + 1) / 2)))).subs(
        {epsilon: epsilon, d: d}))

    # Apply randomized mechanism
    Perturbed_all_kv = cob.CSKV_process(PS, e1, e2, alpha, beta, gamma)
    print("Perturbation complete.")

    # Aggregate and estimate frequency and mean
    f_k, m_k = cob.CSKV_C_E(n, Perturbed_all_kv, d, gamma, e2)
    print("Aggregation complete.")

    # Sort results by estimated frequency
    est = [(i + 1, f_k[i], m_k[i]) for i in range(d)]
    est.sort(key=lambda x: x[1], reverse=True)

    # Select top-2k candidates for evaluation
    est_f = {}             # Estimated frequency dict
    est_m = {}             # Estimated mean dict
    Candidate_rank = []    # Candidate keys sorted by freq
    for i in range(2 * k):
        est_f[est[i][0]] = est[i][1]
        est_m[est[i][0]] = est[i][2]
        Candidate_rank.append(est[i][0])

    # Load ground truth for evaluation
    np.load.__defaults__ = (None, True, True, 'ASCII')  # Fix numpy load bug for object arrays
    base_path = f'./data/{data_name}/CSKV_{data_name}_'
    true_f = np.load(base_path + 'true_f.npy').item()
    true_m = np.load(base_path + 'true_m.npy').item()
    true_rank_all = cob.readtxt(base_path + 'true_rank.txt')[0]
    true_rank = [int(float(x)) for x in true_rank_all[:2 * k]]

    # Evaluation metrics
    MSE_f = "{:.10f}".format(cob.MSE(Candidate_rank, est_f, true_f, k))              # Frequency MSE
    MSE_m = "{:.10f}".format(cob.MSE_CSKV(Candidate_rank, est_m, true_m, k, gamma))  # Mean MSE
    hit_ratio = "{:.2%}".format(cob.hit_ratio(Candidate_rank, true_rank, k))         # Hit ratio
    NCR = "{:.8f}".format(cob.NCR_CSKV(Candidate_rank, true_rank))                   # Normalized cumulative rank

    # Record and save result
    save_path = 'result/result.txt'
    time_stamp = time.strftime("%Y-%m-%d %X")
    end_time = time.time()
    run_time = end_time - start_time
    formatted_run_time = "{:.2f}".format(run_time)

    result = [code_name, data_name, epsilon, k, MSE_f, MSE_m, hit_ratio, NCR,
              time_stamp, formatted_run_time, pad_ell]
    cob.savetxt(result, save_path)

    return 0

def get_parameters():
    # Define dataset names and privacy settings
    data_names = ['IoT_Gaussian_kv_3_4', 'IoT_Gaussian_kv_3_6', 'IoT_Gaussian_kv_4_6',
                  'IoT_Linear_kv_3_4', 'IoT_Linear_kv_3_6', 'IoT_Linear_kv_4_6',
                  'Credit', 'Medical', 'DataCo', 'Cars']
    eps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    # Dataset-specific gamma and padding
    gamma = {'IoT_Gaussian_kv_3_4': 100, 'IoT_Gaussian_kv_3_6': 100, 'IoT_Gaussian_kv_4_6': 100,
             'IoT_Linear_kv_3_4': 100, 'IoT_Linear_kv_3_6': 100, 'IoT_Linear_kv_4_6': 100,
             'Credit': 28.99, 'Medical': 10, 'DataCo': 6, 'Cars': 5}
    pad_ell = {'IoT_Gaussian_kv_3_4': 1, 'IoT_Gaussian_kv_3_6': 1, 'IoT_Gaussian_kv_4_6': 1,
               'IoT_Linear_kv_3_4': 1, 'IoT_Linear_kv_3_6': 1, 'IoT_Linear_kv_4_6': 1,
               'Credit': 1, 'Medical': 2, 'DataCo': 2, 'Cars': 5}

    # Generate parameter combinations
    parameters = []
    for name in data_names[0:9]:
        for e in eps:
            parameters.append([name, e, 0.20, gamma[name], pad_ell[name]])
    return parameters

if __name__ == '__main__':
    # Disable scientific notation in output
    np.set_printoptions(suppress=True)

    # Run CSKV algorithm for all parameter settings
    parameters = get_parameters()
    print(parameters)
    for param in parameters:
        for i in range(3):
            print("Executing, data:", param[0], "eps:", param[1], "round", i + 1)
            CSKV(param)
