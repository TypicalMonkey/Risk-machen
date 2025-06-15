
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

def generate_goe_matrix(N):
    A = np.random.normal(0, 1, size=(N, N))
    A = (A + A.T) / 2
    return A / np.sqrt(N)

def get_normalized_spacings(N, samples):
    spacings = []
    for _ in range(samples):
        M = generate_goe_matrix(N)
        eigenvalues = np.sort(eigvalsh(M))
        center = N // 2
        for k in range(center - 1, center + 2):
            s = eigenvalues[k+1] - eigenvalues[k]
            spacings.append(s)
    spacings = np.array(spacings)
    return spacings / np.mean(spacings)

N = 16
samples = 10000
spacings = get_normalized_spacings(N, samples)

plt.figure(figsize=(8, 4))
plt.hist(spacings, bins=100, density=True, alpha=0.6, label='Numerical spacings')
s_vals = np.linspace(0, 4, 500)
p_wigner = (np.pi / 2) * s_vals * np.exp(- (np.pi / 4) * s_vals**2)
plt.plot(s_vals, p_wigner, 'r--', label="Wigner's surmise")

plt.xlabel("Normalized spacing s")
plt.ylabel("Probability density")
plt.title("GOE Level Spacings vs Wigner's Surmise")
plt.legend()
plt.tight_layout()
plt.show()
