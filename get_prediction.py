"""
The input data must be sorted according to:
Virial mass, Concentration, Spin, Age, Overdensity,
as a pandas.DataFrame with columns named as: M_h, C_h, S_h, z_h, Delta3_h.
"""

from NF import NF
from NNgauss import NNgauss
from NNclass import NNclass
from missing_features import check_input_data

import pandas as pd
import matplotlib.pyplot as plt

# Load data
input_data = pd.read_csv('subtest_tng300_inputs_missing_Delta3_h.csv')
input_data = check_input_data(input_data)


# Galaxy properties
target_props = ['smass', 'color', 'sSFR', 'radius']

# Models
nngauss = NNgauss(target_props, trial=45)
nf = NF(target_props, trial=99)
nnclass = NNclass(target_props, trial=77)

nngauss_pred = nngauss.get_sample(input_data, n_samples=10)
nf_pred = nf.get_sample(input_data, n_samples=10)
nnclass_pred = nnclass.get_sample(input_data, 5, 2)

print(nngauss_pred.shape)
print(nf_pred.shape)


# Example: plot one sample of stellar mass
# Get true values
target_dataset = pd.read_csv('subtest_tng300_targets.csv')
ytrue = target_dataset['smass'].to_numpy()

# Get the first sample of stellar mass (index 0)
nngauss_ypred = nngauss_pred[0, :, 0]
nf_ypred = nf_pred[0, :, 0]
nnclass_ypred = nnclass_pred[0, :, 0]

# Plot
plt.figure(figsize=(4, 4))
plt.plot([min(ytrue), max(ytrue)], [min(ytrue), max(ytrue)], ls='--', color='k')

plt.scatter(ytrue, nngauss_ypred, color=nngauss.color, label=nngauss.label, marker=nngauss.marker, alpha=0.5)
plt.scatter(ytrue, nf_ypred, color=nf.color, label=nf.label, marker=nf.marker, alpha=0.5)
plt.scatter(ytrue, nnclass_ypred, color=nnclass.color, label=nnclass.label, marker=nnclass.marker, alpha=0.5)

plt.ylabel('Predicted')
plt.xlabel('True')
plt.tight_layout()
plt.show()
