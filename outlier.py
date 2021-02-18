import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

d = 3
n = 500
m = 10 # outliers
p = 5 # steps
x = np.arange(n)

#x = np.linspace(-2 * np.pi, 2 * np. pi, n)
fig, axs = plt.subplots(d, sharex=True)
fig.suptitle('Time Series Outlier Detection')
y_idx = np.random.choice(n - 1, p - 1)
y_diff = np.zeros((n - 1, d))
y_diff[y_idx, :] = np.random.randn(p - 1, d)
y_init = np.random.randn(d)
y = np.concatenate([y_init[None, :], y_init[None, :] + np.cumsum(y_diff, axis=0)], axis=0)
y = np.cumsum(y, axis=0)
print(y.shape)
# y1 = np.sin(x)
# y2 = np.cos(2 * x)
z_idx = np.random.choice(n, m)
y_noisy = y.copy()
y_noisy[z_idx, :] += 30 * np.random.randn(m, d)
for i in range(d):
    axs[i].plot(x, y_noisy[:, i], label="Noisy")
    axs[i].plot(x, y[:, i], 'r-', label="Original")
    
    #for j in z_idx:
        #axs[i].plot(x[j-1:j+2], y[j-1:j+2, i], 'r-')
plt.xlabel("Time [Index]")
axs[1].set(ylabel="Feature [Dimensionless]")
axs[0].legend(loc="upper right")
plt.show()
fig.savefig("outlier2.pdf", dpi=600)

