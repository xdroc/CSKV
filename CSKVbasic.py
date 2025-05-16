import numpy as np
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


# Randomly select one element from candidate list with given probability distribution
def mpp(candidate: list, p: list):
    return np.random.choice(candidate, p=p)


# Pre-processing step: padding and candidate selection
def PPS(k_v, d: int, label, pad_ell, delta):
    # Sample labels not in user's key list
    missing_labels = set(label) - set([item[0] for item in k_v])
    s_ = [(i, 0) for i in random.sample(list(missing_labels),
                                        min(pad_ell, len(missing_labels)))]

    # Compute selection probability
    rou = delta * len(k_v) / (delta * len(k_v) + pad_ell)
    rnd = np.random.random()
    chosen_item = random.sample(k_v if rnd < rou else s_, 1)[0]

    # Final padded list
    S_P = k_v + s_
    return chosen_item[0], chosen_item[1], S_P


# Core perturbation process for all users
def CSKV_process(all_kv, e1, e2, alpha, belta, gamma):
    def CSKV(PS, e1, e2, alpha, beta, gamma):
        key, value, S_P = PS
        S_P_key = [item[0] for item in S_P]

        # Compute distance-based weights for key perturbation
        d_s = [1 / (1 + math.exp(- alpha * (abs(i - key) - beta))) for i in
               S_P_key]
        mol = {i: math.exp(-(e1 * d) / 2) for i, d in zip(S_P_key, d_s)}
        theta_k = sum(mol.values())
        pr = {key: value / theta_k for key, value in mol.items()}

        # Randomly select a key from S_P
        selected_key = \
        random.choices(list(pr.keys()), weights=pr.values(), k=1)[0]

        # Value perturbation with symmetric interval
        T = 8 * gamma / (math.exp(e2) - 1)
        theta_v = math.exp(e2) / (math.exp(e2) + 1)
        L = (1 + 2 / math.exp(e2)) * value - 4 * gamma / (math.exp(e2) - 1)
        R = (1 + 2 / math.exp(e2)) * value + 4 * gamma / (math.exp(e2) - 1)

        v_in = random.uniform(L, R)  # Biased (true) value
        v_out = random.uniform(-T,
                               L) if random.random() < 0.5 else random.uniform(
            R, T)  # Noise value

        # Perturbed output
        if selected_key == key:
            k_p = key
            v_p = mpp([v_in, v_out], [theta_v, 1 - theta_v])
        else:
            k_p = selected_key
            v_p = mpp([v_in, v_out], [0.5, 0.5])

        return (k_p, v_p)

    # Apply CSKV to all users
    return [CSKV(x, e1, e2, alpha, belta, gamma) for x in all_kv]


# Clamp a value within [low, high] range
def correct(low, high, value):
    if value < low:
        return low
    elif value > high:
        return high
    return value


# Aggregate perturbed key-value pairs to estimate frequency and mean
def CSKV_C_E(n, all_kv, d, gamma, e2):
    # Frequency estimation
    k_counter = Counter(kv[0] for kv in all_kv)
    k_num = [k_counter[i + 1] for i in range(d)]
    f_k_perturbed = [count / n for count in k_num]

    # Mean estimation per key using multithreading
    def calculate_mean_for_key(key):
        values = [kv[1] for kv in all_kv if kv[0] == key]
        return np.mean(values) if values else 0

    with ThreadPoolExecutor() as executor:
        m_k_perturbed = list(
            executor.map(calculate_mean_for_key, range(1, d + 1)))

        # Apply correction for means only when frequency > 0
        def calculate_f_k_m_k(f_k_val, m_k_val):
            if f_k_val != 0:
                f_k_k = f_k_val
                m_k_k = correct(1, gamma, m_k_val)
                return f_k_k, m_k_k
            return 0, 0

        f_k, m_k = zip(*[calculate_f_k_m_k(f_k_val, m_k_val)
                         for f_k_val, m_k_val in
                         zip(f_k_perturbed, m_k_perturbed)])
        return list(f_k), list(m_k)


# Mean Squared Error for frequency estimation
def MSE(candidate, est, true, topk):
    square = sum(np.square(est[k] - true[k]) for k in candidate[:topk])
    return float('{:.9f}'.format(square / topk))


# Mean Squared Error for value estimation (normalized by gamma)
def MSE_CSKV(candidate, est, true, topk, gamma):
    square = sum(np.square(est[k] - true[k]) for k in candidate[:topk])
    res = square / topk
    res_new = (2 / gamma) ** 2 * res
    return float('{:.9f}'.format(res_new))


# Hit ratio for top-k set
def hit_ratio(l1, l2, topk):
    return len(set(l1[:topk]).intersection(set(l2[:topk]))) / topk


# Quality score used in NCR (Normalized Cumulative Rank)
def Q(key, true_keys):
    if key not in true_keys:
        return 0
    return float(len(true_keys) - true_keys.index(key))


# Normalized Cumulative Rank for ranking quality evaluation
def NCR_CSKV(Candidate_rank, true_rank):
    true_key_set = set(true_rank)
    numerator = sum(
        Q(key, true_rank) for key in Candidate_rank if key in true_key_set)
    denominator = sum(Q(key, true_rank) for key in true_rank)
    return numerator / denominator if denominator > 0 else 0.0


# Read text file into list of lists, supporting multiple types
def readtxt(path, numtype='float'):
    data_list = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            items = re.split(r'[,\s;| ]+', line.strip())
            if numtype == 'float':
                num = list(map(float, items))
            elif numtype == 'int':
                num = list(map(int, items))
            elif numtype == 'str':
                num = items
            else:
                raise ValueError(f"Unsupported numtype: {numtype}")
            data_list.append(num)
            line = f.readline()
    return data_list


# Save list of values to a text file with optional newline
def savetxt(data: list, path: str, flag: bool = True):
    with open(path, 'a') as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '').replace('(',
                                                                       '').replace(
                ')', '')
            s = s.replace("'", '').replace(',', ' ')
            s += '\t'
            file.write(s)
        if flag:
            file.write('\n')
    print("File saved successfully")
