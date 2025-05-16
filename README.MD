# Achieving Privacy-Preserving and High-Accuracy Collection of Key-Value Data with Local Differential Privacy

This repository contains a Python implementation of the **CSKV** (Cohesive and Segmented Key-Value Data Collection) scheme for collecting and analyzing key-value pair data under Local Differential Privacy (LDP). It supports privacy-preserving frequency and mean estimation in scenarios where users' data are represented as key-value pairs.

## Repository Structure

- `CSKV.py`: Entry point for running experiments, includes parameter settings, data reading, and performance evaluation across various privacy budgets.
- `CSKVbasic.py`: Core library that implements key procedures including:
  - Pre-processing and padding
  - Key perturbation
  - Value perturbation
  - Aggregation (frequency and mean estimation)
  - Evaluation metrics (e.g., MSE, Hit Ratio, NCR)

## Getting Started

### Requirements

- Python 3.6+
- NumPy
- Scikit-learn
- tqdm (optional, for progress bars)

Install dependencies (if using a virtual environment):

```sh
pip install numpy scikit-learn tqdm
```


### Evaluation Metrics

The following metrics are used for assessing the utility of the mechanism:

- `MSE`: Mean Squared Error for frequency estimation
- `MSE_CSKV`: MSE for value estimation normalized by gamma
- `Hit Ratio`: Overlap in top-k estimated vs. true keys
- `NCR`: Normalized Cumulative Rank, measuring ranking quality

## Privacy Model

The mechanism ensures **ε-LDP** for both the key and value domains by introducing calibrated noise:

- Keys are selected based on a probability distribution sensitive to key distances.

- Values are perturbed within a symmetric interval determined.

## Citation

If you use this code in your research, please consider citing the associated paper (to be updated once published).
