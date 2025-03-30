import scipy.stats as stats

# --- Case 1 ---
# Normal distribution N(µ, σ) with µ = 2, σ = 0.3
mu1 = 2
sigma1 = 0.3
quantile_prob1 = 0.9  # We want the 0.9 quantile

# Calculate the 0.9 quantile
# The ppf (Percent Point Function) is the inverse of the CDF (Cumulative Distribution Function)
# It gives the value x such that P(X <= x) = quantile_prob
quantile_value1 = stats.norm.ppf(quantile_prob1, loc=mu1, scale=sigma1)

print(f"--- Normal Distribution N(µ={mu1}, σ={sigma1}) ---")
print(f"The {quantile_prob1} quantile is: {quantile_value1:.4f}")
print(f"Interpretation: 90% of the values from this distribution are expected to be less than or equal to {quantile_value1:.4f}\n")


# --- Case 2 ---
# Normal distribution N(µ, σ) with µ = 100, σ = 6
mu2 = 100
sigma2 = 6
quantile_prob2 = 0.15 # We want the 0.15 quantile

# Calculate the 0.15 quantile
quantile_value2 = stats.norm.ppf(quantile_prob2, loc=mu2, scale=sigma2)

print(f"--- Normal Distribution N(µ={mu2}, σ={sigma2}) ---")
print(f"The {quantile_prob2} quantile is: {quantile_value2:.4f}")
print(f"Interpretation: 15% of the values from this distribution are expected to be less than or equal to {quantile_value2:.4f}")