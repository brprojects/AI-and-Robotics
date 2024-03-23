from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

def target_distribution(x):
    return 0.5 * norm.pdf(x, 20, 3) + 0.5 * norm.pdf(x, 40, 10)

def acceptance_prob(proposed, current):
    ratio1 = target_distribution(proposed) / target_distribution(current)
    acc_prob = min(1, ratio1)
    return acc_prob

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations, sigma):
    samples = []
    current = 20.0
    # count = 0

    for _ in range(iterations):
        # Randomly choose a value in this Gaussian
        proposed = np.random.normal(loc=current, scale=sigma)

        # Calculate acceptance probability
        acc_prob = acceptance_prob(proposed, current)

        # Accept or reject the proposed sample
        if np.random.rand() < acc_prob:
            current = proposed
            # count += 1
        
        # Store the current sample
        samples.append(current)

    # print(count)
    return samples

# Number of iterations and proposal standard deviation
iterations = 20000
# Want 50-85% acceptance rate
sigma = 10 # sigma = 10 gives like 62%

# Generate samples using Metropolis-Hastings
samples = metropolis_hastings(iterations, sigma)

# Plot the histogram of samples
plt.hist(samples, bins=100, density=True, alpha=0.5, color='b', label='Samples')

# Plot the target distribution
x = np.linspace(0, 80, 1000)
plt.plot(x, target_distribution(x), 'r-', linewidth=2, label='Target Distribution')

plt.show()
# plt.savefig('../images/metropolis_hastings_sampling.png')

