from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

def target_distribution(x):
    return 0.5 * norm.pdf(x, 20, 3) + 0.5 * norm.pdf(x, 40, 10)

# Rejection sampling algorithm
def rejection_sampling(iterations):
    samples = []
    max_target = 0.07

    for _ in range(iterations):
        # Sample from the proposal distribution (uniform distribution)
        x = np.random.uniform(low=0, high=80)
        
        # Calculate acceptance probability
        acc_prob = target_distribution(x) / max_target

        # Accept or reject the sample based on acceptance probability
        if np.random.rand() < acc_prob:
            samples.append(x)

    return samples

# Number of iterations and proposal standard deviation
iterations = 20000

# Generate samples using Metropolis-Hastings
samples = rejection_sampling(iterations)
print(len(samples))
# Plot the histogram of samples
plt.hist(samples, bins=100, density=True, alpha=0.5, color='b', label='Samples')

# Plot the target distribution
x = np.linspace(0, 80, 1000)
plt.plot(x, target_distribution(x), 'r-', linewidth=2, label='Target Distribution')

# plt.show()
plt.savefig('../images/rejection_sampling.png')
