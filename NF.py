# Libraries
from parameter_space import ParameterSpace

import numpy as np
import joblib
import torch
import tensorflow_probability as tfp
import pyro.distributions as dist

tfd = tfp.distributions
tfpl = tfp.layers

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used by PyTorch.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU instead.")
# device = torch.device("cpu")


class NF(ParameterSpace):

    def __init__(self, target_props, trial,
                 model_name='NF', color='firebrick', cmap='Reds',
                 marker='*', label=None):

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
        dir_name = self.dir_name

        y_transform = torch.load(dir_name + f'trial{self.trial}-x2_transform.pt', map_location=device)

        dist_base_y = dist.Normal(torch.zeros(self.Ndims, device=device), torch.ones(self.Ndims, device=device) * 0.2)
        dist_y_given_x = dist.ConditionalTransformedDistribution(dist_base_y, [y_transform])

        return dist_y_given_x

    def get_sample(self, input_data, n_samples=1, batch_size=1000):
        """
        :param input_data: dataset to apply the model
        :param n_samples: number of samples per instance
        :param batch_size: batch size
        :return: predicted sample shape=(n_samples, num_simulations, ndim)
        """

        dist_y = self.get_model()
        y_scaler_loaded = self.y_scaler()
        x_scaler_loaded = self.x_scaler()

        # Prepare input data
        # Scale input data
        input_data_scaled = x_scaler_loaded.transform(input_data)
        # Convert to tensor
        input_data_scaled = torch.FloatTensor(input_data_scaled)
        # Move to device
        input_data_scaled = input_data_scaled.to(device)

        # Initialize variable
        pred_total_scaled = np.zeros((n_samples, len(input_data), self.Ndims))

        for i in range(0, len(input_data), batch_size):
            # Monitor progress
            if i % 100 == 0:
                print('# instance: {}'.format(i))

            # Predict batch
            start, end = i, min(i + batch_size, len(input_data))
            batch_input_data = input_data_scaled[start:end]

            pred_batch = dist_y.condition(batch_input_data).sample(torch.Size([n_samples,
                                                                               len(batch_input_data)]))

            pred_total_scaled[:, start:end, :] = pred_batch.cpu().numpy()

        # Revert scaling
        sample = y_scaler_loaded.inverse_transform(pred_total_scaled.reshape(n_samples * len(input_data), self.Ndims))
        sample = sample.reshape(n_samples, len(input_data), self.Ndims)

        if self.name_of_event_space == 'smass_color_sSFR_radius':
            sample = sample[:, :, [0, 3, 1, 2]]  # hard coded: smass, color, sSFR, radius

        return sample
