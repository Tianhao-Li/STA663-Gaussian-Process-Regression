import pandas as pd
import numpy as np

# Function 1
rng = np.random.default_rng(seed=202202210)

n = 75

x = np.sort(rng.uniform(0, 2, n))

D = np.array( [[(x_i - x_j)**2 for x_i in x] for x_j in x] )


s2_0 = 0.1
s2_1 = 2
l = 1

Sigma = np.diag([s2_0] * n) + s2_1 * np.exp(-D/(2*l))

L = np.linalg.cholesky(Sigma)

y = L @ rng.normal(size=(n,1))

plt.figure()
plt.plot(x,y,".")
plt.show()


pd.DataFrame({"x":x, "y": y.ravel()}).to_csv("data/d1.csv", index=False)


# Function 2
rng = np.random.default_rng(seed=202202211)

n = 100
x = np.sort(rng.uniform(0, 2, n))
y = np.sin(3*np.pi*x) / (3*x+1) + rng.normal(0, 0.1, n)

pd.DataFrame({"x":x, "y": y}).to_csv("data/d2.csv", index=False)


# Function 3
rng = np.random.default_rng(seed=202202212)

n = 50
x = np.sort(rng.uniform(0, 3, n))
y = (x-1)**4-2*(x-1)**3-(x-1)**2+(x-1) + rng.normal(0, 0.1, n)

pd.DataFrame({"x":x, "y": y}).to_csv("data/d3.csv", index=False)

