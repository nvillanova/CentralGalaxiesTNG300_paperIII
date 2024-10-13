"""
For fixed number of classes (fixed likelihood), the optuna score is the loss.
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
print(tf.__version__)

# import tensorflow.keras as keras
# from keras.models import Model
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


tfd = tfp.distributions
tfpl = tfp.layers

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("CPUs Available:", tf.config.list_physical_devices('CPU'))


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def objective(trial):

    # Trial settings
    ntrial = trial.number
    print('trial: ', ntrial)

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
    # x = tfp.layers.DistributionLambda(lambda t: tfd.OneHotCategorical(int(nbin)))(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

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

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(**reduceLR_params),
                 tf.keras.callbacks.EarlyStopping(**earlystop_params),
                 tf.keras.callbacks.ModelCheckpoint(**checkpoint_params),
                 tf.keras.callbacks.CSVLogger(**logger_params)]

    start = time.perf_counter()

    history = model.fit(xtrain, ytrain_cat,
                        batch_size=100,
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


# Settings
target_props = ['smass']
input_props = ['M_h', 'C_h', 'S_h', 'z_h', 'Delta3_h']

n_trials = 1

nnclass = NNclass(target_props)

nbin = 50
hival_object = HiVAl(target_props, nbin=nbin)

# Load data
train = pd.read_csv('data/train_tng300.csv')
val = pd.read_csv('data/val_tng300.csv')

xtrain = train[input_props].to_numpy()
xval = val[input_props].to_numpy()

# ytrain, yval defined only once because nbin is fixed
domains = hival_object.domains()  # contains the domains of each instance
ytrain = domains[:len(xtrain)]
yval = domains[len(xtrain):]
ytrain_cat = tf.keras.utils.to_categorical(ytrain, nbin)
yval_cat = tf.keras.utils.to_categorical(yval, nbin)
print(ytrain_cat.shape)


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
study.optimize(objective, n_trials=n_trials)

print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))

# Save study
joblib.dump(study, f"{dir_name}/study.pkl")
