import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # For Gaussian (Normal) distribution CDF, PDF, PPF

# --- Basic Gaussian Properties (Standard Normal N(0, 1)) ---
mu, sigma = 0, 1

def F(x):
    """CDF of the standard normal distribution."""
    return norm.cdf(x, loc=mu, scale=sigma)

def f(x):
    """PDF of the standard normal distribution."""
    return norm.pdf(x, loc=mu, scale=sigma)

# --- (a) Exact Distribution of the Maximum ---

def cdf_max(x, N):
    """Exact CDF of the maximum of N i.i.d. N(0, 1) variables."""
    Fx = F(x)
    # Handle cases where F(x) is exactly 0 or 1 for numerical stability
    # F(x)^N can be 0 if F(x)=0. For F(x)=1, F(x)^N = 1.
    # Use np.power for element-wise exponentiation
    return np.power(Fx, N)

def pdf_max(x, N):
    """Exact PDF of the maximum of N i.i.d. N(0, 1) variables."""
    # PDF_max(x) = d/dx [F(x)^N] = N * [F(x)]^(N-1) * f(x)
    Fx = F(x)
    fx = f(x)

    # Numerical stability for Fx close to 0 or 1, especially when N is large.
    # If Fx is very small (e.g., < 1e-10), Fx^(N-1) might underflow or give NaN.
    # If Fx is exactly 0, the PDF should be 0.
    # If Fx is exactly 1, f(x) will be practically 0, so PDF is 0.
    pdf_val = np.zeros_like(x, dtype=float)
    # Calculate only where Fx is reasonably > 0 to avoid log(0) or 0^large_power issues
    valid_mask = (Fx > 1e-15) & (Fx < 1.0) # Avoid exact 0 or 1

    # Using exp(log(...)) can be more stable than direct power for large N-1
    # N * exp((N - 1) * log(Fx)) * f(x)
    pdf_val[valid_mask] = N * np.power(Fx[valid_mask], N - 1) * fx[valid_mask]
    # Handle edge case N=1 separately if needed, though formula works (N * Fx^0 * fx)
    if N==1:
         return fx # Max of 1 is just the variable itself

    return pdf_val

# --- (b) Gumbel Approximation ---

def calculate_aN(N):
    """Calculate the Gumbel location parameter a_N."""
    # aN = F^{-1}(1 - 1/N)
    # Use Percent Point Function (PPF), which is the inverse CDF
    if N <= 1: # Handle N=1 case somewhat reasonably if needed
        return norm.ppf(0.5, loc=mu, scale=sigma) # Median
    # Ensure argument is strictly < 1 for ppf
    prob = min(1.0 - 1.0/N, 1.0 - 1e-16)
    return norm.ppf(prob, loc=mu, scale=sigma)

def calculate_bN(N, aN):
    """Calculate the Gumbel scale parameter b_N."""
    # bN = F^{-1}(1 - 1/(N*e)) - aN
    if N <= 1:
        # Define somewhat arbitrarily for N=1, e.g., Interquartile range
        return norm.ppf(0.75) - norm.ppf(0.25)
    # Ensure argument is strictly < 1 for ppf
    prob = min(1.0 - 1.0/(N * np.e), 1.0 - 1e-16)
    return norm.ppf(prob, loc=mu, scale=sigma) - aN

def gumbel_pdf_standard(u):
    """Standard Gumbel PDF (for variable u)."""
    # PDF = exp(-u - exp(-u))
    return np.exp(-u - np.exp(-u))

def gumbel_pdf_rescaled(x, aN, bN):
    """Rescaled Gumbel PDF expressed as a function of x."""
    # Transformation: u = (x - aN) / bN
    # p(x) = p_gumbel(u) * |du/dx| = p_gumbel((x - aN) / bN) * (1 / bN)
    if bN <= 1e-15: # Avoid division by zero if bN is somehow non-positive/tiny
        return np.zeros_like(x)
    u = (x - aN) / bN
    return (1.0 / bN) * gumbel_pdf_standard(u)

# --- Plotting ---

N_values = [10, 100, 1000, 10000]
# Determine a suitable x-range dynamically based on largest N
N_max = N_values[-1]
aN_max = calculate_aN(N_max)
bN_max = calculate_bN(N_max, aN_max)
# Range covers roughly mean +/- several standard deviations of the Gumbel approx
x_min = aN_max - 6 * bN_max
x_max = aN_max + 10 * bN_max
# Adjust x_min manually if it goes too low (e.g. below 0 for N=10 looks odd)
x_min = max(0, x_min) # Let's start from 0 as max of N(0,1) is likely positive for N>1
print(f"Plotting range estimated from N={N_max}: x = [{x_min:.2f}, {x_max:.2f}]")
x_plot = np.linspace(x_min, x_max, 1000)

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values))) # Color map for different N

print("\nCalculated Gumbel Parameters:")
print("----------------------------")
print(f"{'N':>6} | {'a_N':>10} | {'b_N':>10}")
print("----------------------------")

for i, N in enumerate(N_values):
    # Calculate exact PDF of max
    y_exact = pdf_max(x_plot, N)

    # Calculate Gumbel parameters
    aN = calculate_aN(N)
    bN = calculate_bN(N, aN)
    print(f"{N:6d} | {aN:10.4f} | {bN:10.4f}")

    # Calculate rescaled Gumbel PDF
    y_gumbel = gumbel_pdf_rescaled(x_plot, aN, bN)

    # Plot both
    plt.plot(x_plot, y_exact, label=f'Exact Max PDF (N={N})', color=colors[i], linestyle='-', linewidth=2)
    plt.plot(x_plot, y_gumbel, label=f'Gumbel Approx (N={N})', color=colors[i], linestyle='--', linewidth=1.5)

print("----------------------------")

plt.title('PDF of Max of N i.i.d. N(0,1) Variables and Gumbel Approximation')
plt.xlabel('x (Value of the Maximum, $X_{max}$)')
plt.ylabel('Probability Density Function (PDF)')
plt.legend(fontsize='small')
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(bottom=0)
# Optional: adjust ylim if peaks become very high for large N
#ymax_overall = max(pdf_max(calculate_aN(N_max), N_max), gumbel_pdf_rescaled(calculate_aN(N_max), calculate_aN(N_max), calculate_bN(N_max, calculate_aN(N_max))))
#plt.ylim(top=ymax_overall * 1.1) # Adjust ylim based on highest peak roughly
plt.tight_layout()
plt.show()

# --- Optional: Plotting CDFs instead (as per NOTE) ---

# plt.figure(figsize=(12, 8))
# print("\nPlotting CDFs:")
# print("----------------------------")
# print(f"{'N':>6} | {'a_N':>10} | {'b_N':>10}")
# print("----------------------------")

# def gumbel_cdf_standard(u):
#      return np.exp(-np.exp(-u))

# def gumbel_cdf_rescaled(x, aN, bN):
#     if bN <= 1e-15: return np.where(x < aN, 0.0, 1.0) # Step function if bN=0
#     u = (x - aN) / bN
#     return gumbel_cdf_standard(u)

# for i, N in enumerate(N_values):
#     # Calculate exact CDF of max
#     y_exact_cdf = cdf_max(x_plot, N)

#     # Calculate Gumbel parameters
#     aN = calculate_aN(N)
#     bN = calculate_bN(N, aN)
#     print(f"{N:6d} | {aN:10.4f} | {bN:10.4f}")

#     # Calculate rescaled Gumbel CDF
#     y_gumbel_cdf = gumbel_cdf_rescaled(x_plot, aN, bN)

#     # Plot both
#     plt.plot(x_plot, y_exact_cdf, label=f'Exact Max CDF (N={N})', color=colors[i], linestyle='-', linewidth=2)
#     plt.plot(x_plot, y_gumbel_cdf, label=f'Gumbel Approx CDF (N={N})', color=colors[i], linestyle='--', linewidth=1.5)

# print("----------------------------")
# plt.title('CDF of Max of N i.i.d. N(0,1) Variables and Gumbel Approximation')
# plt.xlabel('x (Value of the Maximum, $X_{max}$)')
# plt.ylabel('Cumulative Distribution Function (CDF)')
# plt.legend(fontsize='small')
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.ylim([-0.05, 1.05])
# plt.tight_layout()
# plt.show()