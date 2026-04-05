import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
X = np.random.randn(5,3) #create random data matrix
U, S, VT = np.linalg.svd(X, full_matrices=True) #FULL SVD
Uhat, Shat, Vhat = np.linalg.svd(X, full_matrices=False)  #SVD without full matrices



# reading image
A = imread('../Data/grayscale.jpg')
X = np.mean(A, -1) #convert to grayscale
img = plt.imshow(X)

#SVD of image
U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

#approximation of image
for r in (5, 20, 100):
    Xapprox = U[:, :r] @ S[:r, :r] @ VT[:r, :]
    img = plt.imshow(Xapprox)
    plt.show()

#plot singular values and cumulative sum
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

r_values = [5, 20, 100]

ax1 = axes[0]
ax1.semilogy(np.diag(S), color='steelblue', linewidth=1.5)
for r in r_values:
    ax1.axvline(x=r, color='red', linestyle='--', alpha=0.7)
    ax1.plot(r, np.diag(S)[r], 'ro', markersize=8, label=f'r={r}')
ax1.set_xlabel('Singular waarde index')
ax1.set_ylabel('Singuliere waarden (log schaal)')
ax1.set_title('Singuliere waarden')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

cumsum_norm = np.cumsum(np.diag(S)) / np.sum(np.diag(S))
ax2 = axes[1]
ax2.plot(cumsum_norm, color='steelblue', linewidth=1.5)
for r in r_values:
    ax2.axvline(x=r, color='red', linestyle='--', alpha=0.7)
    ax2.plot(r, cumsum_norm[r], 'ro', markersize=8, label=f'r={r} ({cumsum_norm[r]:.1%})')
ax2.set_xlabel('Aantal singuliere waarden (r)')
ax2.set_ylabel('Cumulatieve energie fractie')
ax2.set_title('Cumulatieve som singuliere waarden')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()