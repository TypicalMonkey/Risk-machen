import numpy as np
from scipy.integrate import quad
import warnings

# Parameters for the lognormal distribution
mu = 0  # Since x0 = exp(mu) = 1
sigma = 1

# 1. Define the Lognormal PDF P*LN(x)
def lognormal_pdf(x, mu, sigma):
    """Calculates the lognormal PDF value(s)."""
    if np.any(x <= 0):
        # PDF is zero for x <= 0. Handle to avoid log(0) errors.
        # Create an array of zeros with the same shape as x
        pdf_vals = np.zeros_like(x, dtype=float)
        # Calculate PDF only for x > 0
        mask = x > 0
        pdf_vals[mask] = (1 / (x[mask] * sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-(np.log(x[mask]) - mu)**2 / (2 * sigma**2)))
        return pdf_vals
    else:
        return (1 / (x * sigma * np.sqrt(2 * np.pi)) *
                np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)))

# 2. Define the Perturbed PDF P(x)
def perturbed_pdf(x, mu, sigma, epsilon):
    """Calculates the perturbed PDF value(s)."""
    pln_x = lognormal_pdf(x, mu, sigma)
    # Ensure we only calculate perturbation where x > 0
    perturbation = np.zeros_like(x, dtype=float)
    mask = x > 0
    if epsilon != 0: # Avoid calculating sin if epsilon is 0
         perturbation[mask] = epsilon * np.sin(2 * np.pi * np.log(x[mask]))
    
    # The distribution should still be non-negative.
    # For |epsilon| <= 1, 1 + epsilon*sin(...) is theoretically >= 0.
    # Numerically, let's clamp very small negative values due to precision issues if needed,
    # although for quad it might not be strictly necessary.
    result = pln_x * (1 + perturbation)
    # result[result < 0] = 0 # Optional clamping if strict positivity is needed
    return result

# 3. Define function to calculate the n-th moment numerically
def calculate_numerical_moment(pdf_func, n, *args):
    """
    Calculates the n-th moment E[X^n] = integral x^n * pdf(x) dx from 0 to inf.
    *args are the parameters for the pdf_func (e.g., mu, sigma, epsilon).
    """
    integrand = lambda x: (x**n) * pdf_func(x, *args)

    # Use quad for numerical integration from 0 to infinity
    # Ignore potential integration warnings, especially for higher moments
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moment_val, abs_error = quad(integrand, 0, np.inf, limit=200) # Increase limit for potentially tricky integrals

    return moment_val, abs_error

# 4. Analytical moments for Lognormal(mu, sigma)
def analytical_lognormal_moment(n, mu, sigma):
    """Calculates the analytical n-th moment of the lognormal distribution."""
    return np.exp(n * mu + (n**2 * sigma**2) / 2)

# --- Simulation ---
moment_orders = [0, 1, 2, 3, 4] # Test moments n=0, 1, 2, 3, 4
epsilon_values = [0.0, 0.5, 1.0, -0.5, -1.0] # Test different epsilon values

print(f"Parameters: mu={mu}, sigma={sigma}")
print("-" * 60)
print(f"{'n':<3} | {'Epsilon':<8} | {'Analytical M_n':<18} | {'Numerical P*_LN M_n':<22} | {'Numerical P(x) M_n':<20}")
print("-" * 60)

results_match = True

for n in moment_orders:
    # Calculate the analytical moment for lognormal
    m_analytical = analytical_lognormal_moment(n, mu, sigma)

    # Calculate numerical moment for the base lognormal (equivalent to epsilon=0)
    m_numerical_pln, err_pln = calculate_numerical_moment(lognormal_pdf, n, mu, sigma)

    print(f"{n:<3} | {'N/A':<8} | {m_analytical:<18.6e} | {m_numerical_pln:<22.6e} (err={err_pln:.1e}) | {'-':<20}")

    for eps in epsilon_values:
         # Calculate numerical moment for the perturbed distribution P(x)
        m_numerical_p, err_p = calculate_numerical_moment(perturbed_pdf, n, mu, sigma, eps)

        # Check if the moment for P(x) matches the analytical/base lognormal moment
        # Use relative tolerance for comparison due to potentially large moment values
        if not np.isclose(m_numerical_p, m_analytical, rtol=1e-5):
             results_match = False
             match_status = "<- MISMATCH"
        else:
             match_status = ""

        print(f"{n:<3} | {eps:<8.2f} | {'-':<18} | {'-':<22} | {m_numerical_p:<20.6e} (err={err_p:.1e}) {match_status}")

    print("-" * 60)


# --- Conclusion ---
print("\nConclusion:")
if results_match:
    print("The numerical moments for P(x) match the analytical moments of P*_LN(x)")
    print("across different values of epsilon, within numerical integration tolerance.")
    print("This supports the claim that the moments are independent of epsilon.")
else:
    print("Potential mismatch found (check output above). This could be due to:")
    print("  - Numerical integration errors (especially for higher n).")
    print("  - A mistake in the formulas or code.")
    print("  - The theoretical result might require more careful numerical treatment.")

print("\nNote: Numerical integration accuracy can decrease for higher moments 'n'")
print("because the integrand x^n * P(x) becomes harder to integrate accurately.")

# Optional: Visualize the PDFs to show they are different
try:
    import matplotlib.pyplot as plt

    x_vals = np.linspace(0.01, 10, 500) # Avoid x=0
    pln_vals = lognormal_pdf(x_vals, mu, sigma)
    p_eps_05_vals = perturbed_pdf(x_vals, mu, sigma, 0.5)
    p_eps_10_vals = perturbed_pdf(x_vals, mu, sigma, 1.0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, pln_vals, label='P*_LN(x) (Lognormal, $\\epsilon=0$)')
    plt.plot(x_vals, p_eps_05_vals, label='P(x), $\\epsilon=0.5$', linestyle='--')
    plt.plot(x_vals, p_eps_10_vals, label='P(x), $\\epsilon=1.0$', linestyle=':')
    plt.title('Lognormal vs. Perturbed Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability Density P(x)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.show()
except ImportError:
    print("\nMatplotlib not found. Skipping plot.")