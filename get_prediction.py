from cProfile import label

from NF import NF
from NNgauss import NNgauss

import pandas as pd
import matplotlib.pyplot as plt

# Load data
dataset = pd.read_csv('subtest_tng300.csv')

# Properties
target_props = ['smass', 'color', 'sSFR', 'radius']
input_props = ['M_h', 'C_h', 'S_h', 'z_h', 'Delta3_h']

input_data = dataset[input_props].to_numpy()

# Models
# the output shape is (n_samples, n_simulations, n_dimensions)
nngauss = NNgauss(target_props, trial=45)
nf = NF(target_props, trial=99)

nngauss_pred = nngauss.get_sample(input_data, n_samples=10)
nf_pred = nngauss.get_sample(input_data, n_samples=10)

print(nngauss_pred.shape)
print(nf_pred.shape)


# Example: plot one sample of stellar mass
nngauss_smass = nngauss_pred[0, :, 0]
nf_smass = nf_pred[0, :, 0]

plt.scatter(nngauss_smass, dataset['smass'],
            color=nngauss.color, label=nngauss.label, marker=nngauss.marker,
            alpha=0.5)
plt.scatter(nf_smass, dataset['smass'],
            color=nf.color, label=nf.label, marker=nf.marker,
            alpha=0.5)

plt.ylabel('Predicted')
plt.xlabel('True')
plt.show()
