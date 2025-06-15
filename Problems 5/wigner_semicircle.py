def generate_wigner_matrix(n, sampler):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                M[i, j] = sampler()
            else:
                val = sampler()
                M[i, j] = val
                M[j, i] = val
    return M / np.sqrt(n)

def empirical_spectral_distribution(eigenvalues, bins=100):
    hist, bin_edges = np.histogram(eigenvalues, bins=bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, hist

def semicircle_density(x, sigma=1):
    return (1 / (2 * np.pi * sigma**2)) * np.sqrt(np.maximum(0, 4 * sigma**2 - x**2))

n = 1000
sampler = lambda: np.random.uniform(-1, 1)
W = generate_wigner_matrix(n, sampler)
eigenvalues = eigvalsh(W)


x_empirical, y_empirical = empirical_spectral_distribution(eigenvalues, bins=150)
x_vals = np.linspace(-2, 2, 1000)
y_semicircle = semicircle_density(x_vals, sigma=np.sqrt(1/3))#Var[uniform(-1,1)]=1/3

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_semicircle, color='red', label='Semicircle Law')
plt.bar(x_empirical, y_empirical, width=(x_empirical[1] - x_empirical[0]), alpha=0.6, label='Empirical ESD')
plt.legend()
plt.title("Wigner Semicircle Law vs Empirical Spectral Distribution")
plt.xlabel("Eigenvalue")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
