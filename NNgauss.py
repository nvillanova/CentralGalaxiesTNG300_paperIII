# Libraries
from parameter_space import ParameterSpace

import pandas as pd
import joblib
import tensorflow as tf
from keras import Model
from keras.layers import Dense
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class NNgauss(ParameterSpace):

    def __init__(self, target_props, trial,
                 model_name='NNgauss', color='yellowgreen', cmap='GrBu',
                 marker='o', label=None):

        super().__init__(target_props)

        self.trial = trial

        self.model_name = model_name
        self.color = color
        self.cmap = cmap
        self.marker = marker
        self.label = label
        if self.label is None:
            self.label = model_name

        self.dir_name = 'models/{}/{}/'.format(self.model_name,
                                        self.name_of_event_space)

    def get_model_dict(self):
        dir_name = self.dir_name

        # Load model dictionary
        model_dict = pd.read_csv(dir_name + 'trial_dict_{}.csv'.format(self.trial))
        return model_dict

    def x_scaler(self):

        dir_name = self.dir_name

        x_scaler_filename = dir_name + "X_scaler.save"
        x_scaler_loaded = joblib.load(x_scaler_filename)

        return x_scaler_loaded

    def y_scaler(self):
        dir_name = self.dir_name

        y_scaler_filename = dir_name + "Y_scaler.save"
        y_scaler_loaded = joblib.load(y_scaler_filename)

        return y_scaler_loaded

    def get_model(self):
        """
        Load model's weights.
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

        x = Dense(tfpl.MultivariateNormalTriL.params_size(int(self.Ndims)))(x)
        x = tfp.layers.DistributionLambda(lambda t:
                                          tfd.MultivariateNormalTriL(loc=t[..., :int(self.Ndims)],
                                                                     scale_tril=tfp.math.fill_triangular(t[...,
                                                                                                         int(self.Ndims):])))(x)

        loaded_model = Model(inputs=inputs, outputs=x)

        # Load saved weights
        loaded_model.load_weights(self.dir_name + 'weights_trial{}.h5'.format(self.trial))

        return loaded_model

    def get_sample(self, input_data, n_samples=1):

        model = self.get_model()

        x_scaler_loaded = self.x_scaler()
        y_scaler_loaded = self.y_scaler()

        input_data_scaled = x_scaler_loaded.transform(input_data)

        ypred_scaled = model(input_data_scaled).sample(n_samples).numpy()
        ypred = y_scaler_loaded.inverse_transform(ypred_scaled.reshape(n_samples * len(input_data), self.Ndims))
        ypred = ypred.reshape(n_samples, len(input_data), self.Ndims)

        return ypred
