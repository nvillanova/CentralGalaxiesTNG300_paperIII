# Libraries
from parameter_space import ParameterSpace
from hival import HiVAl

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class NNclass(ParameterSpace):

    def __init__(self, target_props, trial=None, nbin=None,
                 model_name='NNclass', color='deepskyblue', cmap='GrBu',
                 marker='v', label=None):

        super().__init__(target_props)

        self.trial = trial
        self.nbin = nbin

        self.model_name = model_name
        self.color = color
        self.cmap = cmap
        self.marker = marker
        self.label = label
        if self.label is None:
            self.label = model_name

        self.dir_name = f'{self.model_name}/{self.name_of_event_space}/'
        if nbin is not None:
            self.dir_name = f'{self.model_name}/{self.name_of_event_space}_{self.nbin}/'

    def get_model_dict(self):
        dir_name = self.dir_name

        # Load model dictionary
        model_dict = pd.read_csv(dir_name + 'trial_dict_{}.csv'.format(self.trial))
        return model_dict

    def input_features(self):
        dir_name = self.dir_name
        return pd.read_csv(dir_name + 'input_props_trial{}.csv'.format(self.trial)).columns.to_numpy()

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
        nbin = model_dict['nbin'].to_numpy()[0]
        input_shape = model_dict['input_shape'].to_numpy()[0]
        inputs = tf.keras.layers.Input(shape=(input_shape,))

        # Input layer
        x = tf.keras.layers.Dense(hidden_units[0],
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.L2(l2=int(hidden_l2[0])))(inputs)
        # Hidden layers
        for nl in range(1, n_layers):
            x = tf.keras.layers.Dense(hidden_units[nl],
                                      activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.L2(l2=int(hidden_l2[nl])))(x)

        x = tf.keras.layers.Dense(int(nbin))(x)

        loaded_model = tf.keras.models.Model(inputs=inputs, outputs=x)

        # Load saved weights
        filename = self.dir_name + 'weights_trial{}.weights.h5'.format(self.trial)
        if not os.path.exists(filename):
            filename = self.dir_name + 'weights_trial{}.h5'.format(self.trial)

        loaded_model.load_weights(filename)

        return loaded_model

    def get_model(self, input_data):

        model = self.load_model_weights()
        predictions = model.predict(input_data)

        # Create the MultivariateNormalTriL distribution with the pred. parameters
        return tfd.OneHotCategorical(logits=predictions)

    def get_sample(self, input_data,
                   num_domain_sample=1, num_values_sample=1):

        """
        total number of samples == n_samples = num_domain_sample * num_values_sample
        """

        predicted_distribution = self.get_model(input_data)
        nbin = self.get_model_dict()['nbin'].to_numpy()[0]
        hival_object = HiVAl(target_props=self.target_props, nbin=nbin)

        if self.Ndims > 1:

            continuous_value_list = []
            for r in range(num_domain_sample):
                # Realization of the predicted distribution, i.e., sample a class
                predicted_class = predicted_distribution.sample(1)
                predicted_class = np.argmax(predicted_class[0], axis=1)
                # Sample continuous value
                continuous_value = hival_object.sample_continuous_values(predicted_class,
                                                                         num_values_sample)

                continuous_value_list.append(continuous_value)

            sample = np.array(continuous_value_list)
            # Join class realizations and continuous values samples
            sample = sample.reshape(sample.shape[0] * sample.shape[1],
                                    sample.shape[2], sample.shape[3])

        # Univariate
        else:
            print('Sampling from uni-variate distribution')

            bin_edges = hival_object.bin_edges().to_numpy()

            continuous_value_list = []
            for r in range(num_domain_sample):
                # Realization of the predicted distribution, i.e., sample a class
                predicted_class = predicted_distribution.sample(1)
                predicted_class = np.argmax(predicted_class[0], axis=1)

                # Sample continuous values
                # a = bin_edges[predicted_class.T[:, :, None]]
                # b = bin_edges[(predicted_class.T + 1)[:, :, None]]
                a = bin_edges[predicted_class]
                b = bin_edges[predicted_class + 1]

                continuous_value = (b - a) * np.random.random_sample((len(input_data), num_values_sample)) + a

                continuous_value_list.append(continuous_value.T)

            sample = np.array(continuous_value_list)
            # Join class realizations and continuous values samples
            sample = sample.reshape(sample.shape[0] * sample.shape[1],
                                    sample.shape[2])
            sample = sample[:, :, None]

        return sample
