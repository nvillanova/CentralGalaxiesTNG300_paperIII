from NF import NF
from NNgauss import NNgauss
# from NNclass import NNclass

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
# nnclass = NNclass(target_props, trial=77)
nf = NF(target_props, trial=99)

nngauss_pred = nngauss.get_sample(input_data, n_samples=10)
nnclass_pred = nnclass.get_sample(input_data, 5, 2)
nf_pred = nf.get_sample(input_data, n_samples=10)

print(nngauss_pred.shape)
print(nf_pred.shape)


# Example: plot one sample of stellar mass
ytrue = dataset['smass']

nngauss_ypred = nngauss_pred[0, :, 0]
nnclass_ypred = nnclass_pred[0, :, 0]
nf_ypred = nf_pred[0, :, 0]


plt.figure(figsize=(4, 4))

plt.plot([min(ytrue), max(ytrue)], [min(ytrue), max(ytrue)], ls='--', color='k')

plt.scatter(ytrue, nngauss_ypred,
            color=nngauss.color, label=nngauss.label, marker=nngauss.marker,
            alpha=0.5)
plt.scatter(ytrue, nnclass_ypred,
            color=nnclass.color, label=nnclass.label, marker=nnclass.marker,
            alpha=0.5)
plt.scatter(ytrue, nf_ypred,
            color=nf.color, label=nf.label, marker=nf.marker,
            alpha=0.5)

plt.ylabel('Predicted')
plt.xlabel('True')
plt.tight_layout()
plt.show()
