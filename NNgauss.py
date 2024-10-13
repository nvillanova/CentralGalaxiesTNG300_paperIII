# Libraries
from parameter_space import ParameterSpace

import os
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class NNgauss(ParameterSpace):

    def __init__(self, target_props, trial,
                 sats=False,
                 model_name='NNgauss', color='yellowgreen', cmap='GrBu',
                 marker='o', label=None):

        super().__init__(target_props)

        self.trial = trial

        self.sats = sats
        self.model_name = model_name
        self.color = color
        self.cmap = cmap
        self.marker = marker
        self.label = label
        if self.label is None:
            self.label = model_name

        self.dir_name = '{}/{}/'.format(self.model_name,
                                        self.name_of_event_space)

        if sats:
            self.dir_name = '{}/{}/sats/'.format(self.model_name,
                                                 self.name_of_event_space)


    def get_model_dict(self):
        dir_name = self.dir_name

        # Load model dictionary
        model_dict = pd.read_csv(dir_name + 'trial_dict_{}.csv'.format(self.trial))
        return model_dict

    def input_features(self):
        dir_name = self.dir_name
        return pd.read_csv(dir_name + 'input_props_trial{}.csv'.format(self.trial)).columns.to_numpy()

    def x_scaler(self):

        dir_name = self.dir_name

        x_scaler_filename = dir_name + f"X_scaler_trial{self.trial}.save"
        if not os.path.exists(x_scaler_filename):
            x_scaler_filename = dir_name + f"X_scaler.save"

        x_scaler_loaded = joblib.load(x_scaler_filename)

        return x_scaler_loaded

    def y_scaler(self):
        dir_name = self.dir_name

        y_scaler_filename = dir_name + "Y_scaler.save"
        y_scaler_loaded = joblib.load(y_scaler_filename)

        return y_scaler_loaded

    def load_model_weights(self):
        """
        Load model's weights.
        Build architecture using the model dictionary and load the weights.
        """

        # Load model dictionary
        model_dict = self.get_model_dict()

        # Number of layers
        n_layers = model_dict['n_layers'].to_numpy()[0]

        # Number of neurons of each hidden layer (input has fixed number)
        hidden_units = []
        # L2 regularization in hidden layers
        hidden_l2 = []
        # Get number of neurons and L2 parameter for the hidden layers
        for nl in range(n_layers):
            n_units = model_dict['n_units'].to_numpy()[nl]
            l2 = model_dict['l2'].to_numpy()[nl]

            hidden_units.append(n_units)
            hidden_l2.append(l2)

        # Model
        input_shape = model_dict['input_shape'].to_numpy()[0]
        inputs = tf.keras.layers.Input(shape=(input_shape,))

        # Input layer
        x = tf.keras.layers.Dense(hidden_units[0],
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.L2(l2=hidden_l2[0]))(inputs)

        # Hidden layers
        for nl in range(1, n_layers):
            x = tf.keras.layers.Dense(hidden_units[nl],
                                      activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.L2(l2=hidden_l2[nl]))(x)

        # Output layer for parameters (loc and scale_tril)
        params_size = tfpl.MultivariateNormalTriL.params_size(int(self.Ndims))
        x = tf.keras.layers.Dense(params_size)(x)

        loaded_model = tf.keras.models.Model(inputs=inputs, outputs=x)

        # Load saved weights
        filename = self.dir_name + 'weights_trial{}.weights.h5'.format(self.trial)
        if not os.path.exists(filename):
            filename = self.dir_name + 'weights_trial{}.h5'.format(self.trial)

        loaded_model.load_weights(filename)

        return loaded_model

    def get_model(self, input_data):

        model = self.load_model_weights()

        # Scale input parameters
        x_scaler_loaded = self.x_scaler()
        input_data_scaled = x_scaler_loaded.transform(input_data)

        # Predict mean and covariance matrix
        predictions = model.predict(input_data_scaled)
        loc = predictions[..., :int(self.Ndims)]
        scale_tril = tfp.math.fill_triangular(predictions[..., int(self.Ndims):])

        # Create the MultivariateNormalTriL distribution with the pred. parameters
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

    def get_log_prob(self, input_data, target_data):
        model = self.get_model(input_data)

        y_scaler_loaded = self.y_scaler()
        target_data_scaled = y_scaler_loaded.transform(target_data)

        return model.log_prob(target_data_scaled).numpy()

    def get_sample(self, input_data, n_samples=1):
        """
        Load model, get scaled sample and revert scale.
        :param input_data: input data to apply model
        :param n_samples: number of samples to draw from the distribution
        :return: predictions (n_samples, n_sims, n_dims)
        """

        model = self.get_model(input_data)
        y_scaler_loaded = self.y_scaler()

        ypred_scaled = model.sample(n_samples).numpy()
        ypred = y_scaler_loaded.inverse_transform(ypred_scaled.reshape(n_samples * len(input_data), self.Ndims))
        ypred = ypred.reshape(n_samples, len(input_data), self.Ndims)

        return ypred
