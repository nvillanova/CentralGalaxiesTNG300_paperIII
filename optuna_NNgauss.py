# Libraries
from NNgauss import NNgauss

import numpy as np
import pandas as pd
import os
import time
import joblib
import optuna

import tensorflow as tf
import tensorflow_probability as tfp

# from keras.models import Sequential, Model
# from keras.layers import Dense, Input, Layer
# from keras.optimizers import Adam
# from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.preprocessing import MinMaxScaler

tfd = tfp.distributions
tfpl = tfp.layers


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Verify available devices
print("Devices available:", tf.config.list_physical_devices())

print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("CPUs Available:", tf.config.list_physical_devices('CPU'))


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def objective(trial):

    # Trial settings
    ntrial = trial.number
    print('trial: ', ntrial)

    # Adam optimizer learning rate
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)

    # Number of layers
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # Number of neurons of each hidden layer (input has fixed number)
    hidden_units = []
    # L2 regularization in hidden layers
    hidden_l2 = []
    l2 = trial.suggest_float('l2', 1e-5, 1e-1, log=True)
    # Get number of neurons and L2 parameter for the hidden layers
    for nl in range(n_layers):
        n_units = trial.suggest_int('n_units_layer{}'.format(nl), 4, 512)
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

    x = tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(int(len(target_props))))(x)
    x = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalTriL(loc=t[..., :int(len(target_props))],
                                                                           scale_tril=tfp.math.fill_triangular(
                                                                               t[..., int(len(target_props)):])))(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    adam_opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    compile_params = {'loss': nll,
                      'optimizer': adam_opt,
                      'metrics': ['mse'],
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
                        'patience': 40,
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

    history = model.fit(xtrain_scaled, ytrain_scaled,
                        batch_size=100,
                        epochs=200,
                        verbose=1,
                        validation_data=(xval_scaled, yval_scaled),
                        callbacks=callbacks)

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    # Save
    # plot_learn_curves(history, elapsed, save=False)
    model.save_weights(dir_name + 'weights_trial{}.h5'.format(ntrial))
    model.save(dir_name + 'model_trial{}.h5'.format(ntrial))
    model.save(dir_name + 'model_trial{}.keras'.format(ntrial))

    # Model dictionary
    trial_dict = {'ntrial': ntrial,
                  'input_shape': xtrain.shape[1],
                  'n_layers': n_layers,
                  'n_units': hidden_units,
                  'l2': hidden_l2}

    pd.DataFrame.from_dict(trial_dict).to_csv(dir_name +
                                              'trial_dict_{}.csv'.format(ntrial), index=False)

    loss_val = -np.mean(model(xval_scaled).log_prob(yval_scaled))
    print(loss_val)

    return loss_val


# Settings
# target_props = ['smass', 'color', 'sSFR', 'radius']
# input_props = ['M_h', 'C_h', 'S_h', 'z_h', 'Delta3_h']

target_props = ['M_h', 'C_h']
input_props = ['smass', 'color']

nngauss = NNgauss(target_props, 0)

n_trials = 1


# Create directory
dir_name = nngauss.dir_name

print('Creating directory..')
if os.path.exists(dir_name):
    print('ignoring: DIRECTORY EXISTS! All the content will be overwritten!')
else:
    os.mkdir(path=dir_name)

while os.path.isfile(dir_name + 'weights_trial{}.h5'.format(nngauss.trial)):
    print('\nTrial {} exists! Trying ntrial = {}.'.format(nngauss.trial, nngauss.trial + 1))
    nngauss.trial = nngauss.trial + 1

print('Trial: ', nngauss.trial)

# Load data
train = pd.read_csv('data/train_tng300.csv')
val = pd.read_csv('data/val_tng300.csv')

xtrain = train[input_props].to_numpy()
xval = val[input_props].to_numpy()

ytrain = train[target_props].to_numpy()
yval = val[target_props].to_numpy()

# Scale variables
Y_scaler = MinMaxScaler(feature_range=(-1., 1.))
ytrain_scaled = Y_scaler.fit_transform(ytrain)
yval_scaled = Y_scaler.transform(yval)

X_scaler = MinMaxScaler(feature_range=(-1., 1.))
xtrain_scaled = X_scaler.fit_transform(xtrain)
xval_scaled = X_scaler.transform(xval)

# Save fitted scaler
X_scaler_filename = dir_name + "X_scaler.save"
joblib.dump(X_scaler, X_scaler_filename)

Y_scaler_filename = dir_name + "Y_scaler.save"
joblib.dump(Y_scaler, Y_scaler_filename)

# Initialize study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)

print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))

# Save study
joblib.dump(study, f"{dir_name}/study.pkl")
