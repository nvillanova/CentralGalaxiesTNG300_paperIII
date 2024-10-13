"""
nbin is an optuna parameter, need to load ytrain, yval for each nbin,
so this is defined inside the objective function.
Missing: review implementation of PCC as optuna score.
Pode usar o yval ja ajustado ou ajustar na hora.
"""

# Libraries
from hival import HiVAl
from NNclass import NNclass

import numpy as np
import pandas as pd
import os
import time
import optuna
import joblib

import tensorflow as tf
import tensorflow_probability as tfp

from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from scipy.stats import pearsonr

tfd = tfp.distributions
tfpl = tfp.layers


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Verify available devices
print("Devices available:", tf.config.list_physical_devices())

print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("CPUs Available:", tf.config.list_physical_devices('CPU'))


# Model
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def objective(trial):

    # Trial settings
    ntrial = trial.number
    print('trial: ', ntrial)

    # Number of classes
    # prior taken from PCC of perfect classes analysis
    nbin = trial.suggest_categorical('nbin', [3439])
    print('Number of classes: ', nbin)

    # Number of layers
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # Number of neurons of each hidden layer (input has fixed number)
    hidden_units = []
    # L2 regularization in hidden layers
    hidden_l2 = []
    # Get number of neurons and L2 parameter for the hidden layers
    for nl in range(n_layers):
        n_units = trial.suggest_int('n_units_layer{}'.format(nl), 4, 512)
        l2 = trial.suggest_float('l2', 1e-5, 1e-1, log=True)

        hidden_units.append(n_units)
        hidden_l2.append(l2)

    # Need to update the Likelihood (HiVAl), therefore ytrain and yval, for each

    # choice of nbin
    hival_object.nbin = nbin
    ytrain = hival_object.domains()  # contains the domains of each instance
    ytrain_cat = to_categorical(ytrain, nbin)

    # Needed to compute PCC
    particles_list = hival_object.get_particles_list()
    positions = hival_object.positions()
    dispersions = hival_object.dispersions()

    yval = hival_object.assign_HiVAl_cell(val)
    yval_aux = -np.ones(len(xval))
    for domain_ind in range(nbin):
        yval_aux[yval[domain_ind][0]] = domain_ind
    yval_cat = to_categorical(yval_aux.astype(int), nbin)

    # Model
    inputs = tf.keras.layers.Input(shape=(xtrain.shape[1],))

    # Input layer
    x = tf.keras.layers.Dense(hidden_units[0],
                              activation='relu',
                              kernel_regularizer=tf.keras.regularizers.L2(l2=hidden_l2[0]))(inputs)
    # Hidden layers
    for nl in range(1, n_layers):
        x = tf.keras.layers.Dense(hidden_units[nl],
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.L2(l2=hidden_l2[nl]))(x)

    x = tf.keras.layers.Dense(int(nbin))(x)
    x = tfpl.OneHotCategorical(int(nbin))(x)

    model = Model(inputs=inputs, outputs=x)

    compile_params = {'loss': nll,
                      'optimizer': 'Adam',
                      'metrics': ['accuracy'],
                      'loss_weights': None,
                      'sample_weight_mode': None,
                      'weighted_metrics': None,
                      'target_tensors': None}

    # Compile model
    model.compile(**compile_params)
    model.summary()

    reduceLR_params = {'monitor': 'val_loss',
                       'factor': 0.2,
                       'patience': 20,
                       'verbose': 1,
                       'mode': 'min',
                       'min_delta': 0.0001,
                       'cooldown': 0,
                       'min_lr': 0}

    earlystop_params = {'monitor': 'val_loss',
                        'patience': 30,
                        'min_delta': 0.0001,
                        'verbose': 1,
                        'mode': 'min',
                        'baseline': None,
                        'restore_best_weights': False}

    checkpoint_params = {'filepath': dir_name + 'best_val_loss_trial{}.h5'.format(ntrial),
                         'monitor': 'val_loss',
                         'mode': 'min',
                         'verbose': 0,
                         'save_best_only': True,
                         'save_weights_only': True,
                         'save_freq': 'epoch'}

    logger_params = {'filename': dir_name + 'epoch_results_trial{}.csv'.format(ntrial),
                     'separator': ',',
                     'append': False}

    callbacks = [ReduceLROnPlateau(**reduceLR_params),
                 EarlyStopping(**earlystop_params),
                 ModelCheckpoint(**checkpoint_params),
                 CSVLogger(**logger_params)]

    start = time.perf_counter()

    history = model.fit(xtrain, ytrain_cat,
                        batch_size=1000,
                        epochs=200,
                        verbose=1,
                        validation_data=(xval, yval_cat),
                        callbacks=callbacks)

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Save
    # plot_learn_curves(history, elapsed, save=False)
    model.save_weights(dir_name + 'weights_trial{}.h5'.format(ntrial))
    model.save(dir_name + 'model_trial{}.h5'.format(ntrial))

    # Model dictionary
    trial_dict = {'ntrial': ntrial,
                  'nbin': nbin,
                  'input_shape': xtrain.shape[1],
                  'n_layers': n_layers,
                  'n_units': hidden_units,
                  'l2': hidden_l2}

    pd.DataFrame.from_dict(trial_dict).to_csv(dir_name +
                                              'trial_dict_{}.csv'.format(ntrial), index=False)

    loss_val = -np.mean(model(xval).log_prob(yval_cat))
    print(loss_val)

    loaded_model = NNclass(target_props, ntrial).get_model()
    loaded_loss = -np.mean(loaded_model(xval).log_prob(yval_cat))
    print(loaded_loss)

    return loss_val

    # # OPTUNA metric: PCC
    # print('\nPCC\n')
    # start_pcc = time.perf_counter()
    #
    # ypred_continuous = nnclass.get_sample(xval, hival_object, 3)
    #
    # pcc = 0
    # for _ in range(len(target_props)):
    #   pcc_sample = np.zeros(3)
    #   for s in range(3):
    #     pcc_sample[s] = pearsonr(yval_continuous[:, _], ypred_continuous[s, :, _])[0]
    #   print(pcc_sample)
    #   pcc = pcc + np.mean(pcc_sample)
    #   print(target_props[_], np.mean(pcc_sample))
    #
    # elapsed_pcc = time.perf_counter() - start_pcc
    # print('\nPCC computation: elapsed %.3f seconds.' % elapsed_pcc)
    #
    # print('\nPCC (sum over target properties): {}'.format(pcc))
    #
    # trial_dict['pcc'] = pcc
    #
    # pd.DataFrame.from_dict(trial_dict).to_csv(dir_name + 'trial_dict_{}.csv'.format(ntrial), index=False)
    #
    # return pcc


# Settings
target_props = ['smass', 'color', 'sSFR', 'radius']
input_props = ['M_h', 'C_h', 'S_h', 'z_h', 'Delta3_h']

nnclass = NNclass(target_props)
hival_object = HiVAl(target_props)

# Load data
train = pd.read_csv('data/train_tng300.csv')
val = pd.read_csv('data/val_tng300.csv')

xtrain = train[input_props].to_numpy()
xval = val[input_props].to_numpy()


# Create directory
dir_name = nnclass.dir_name

print('Creating directory..')
if os.path.exists(dir_name):
    print('ignoring: DIRECTORY EXISTS! All the content will be overwritten!')
else:
    os.mkdir(path=dir_name)

while os.path.isfile(dir_name + 'weights_trial{}.h5'.format(nnclass.trial)):
    print('\nTrial {} exists! Trying ntrial = {}.'.format(nnclass.trial, nnclass.trial + 1))
    nnclass.trial = nnclass.trial + 1

print('Trial: ', nnclass.trial)

# Initialize study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))

# Save study
joblib.dump(study, f"{dir_name}/study.pkl")

