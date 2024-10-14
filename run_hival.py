"""
Example of running HiVAl (on fake data).
"""

# Libraries
import hival
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Generate fake data in a 3D parameter space
np.random.seed(42)
N = 500

data_Normal = np.random.normal(0, 0.75, size=(3, N)).T
data_Uniform = np.random.uniform(2, 4, size=(3, N)).T
data = pd.DataFrame(np.vstack((data_Normal, data_Uniform)), columns=['x', 'y', 'z'])

# Visualize data
plt.figure(figsize=(4, 4))
plt.scatter(data['x'], data['y'], s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2.5, 5)
plt.ylim(-2.5, 5)
plt.show()


# Settings
# suppose we want to run HiVAl in the {x, y} subspace
target_props = ['x', 'y']

# Run
hival.run_HiVAl(data, target_props)
