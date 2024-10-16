"""
Given a data set, completes the data set with eventual missing features.
The missing feature is a random realization of the input data distribution, learned with NF.
"""
# Libraries
import torch
import pyro.distributions as dist
import pandas as pd
import joblib

device = torch.device("cpu")


def check_input_data(input_dataset):

    N = torch.Size([len(input_dataset)])
    input_props_user = input_dataset.columns

    input_props_original = ['M_h', 'C_h', 'S_h', 'z_h', 'Delta3_h']  # The list to check against
    # Find the elements from input_props_original that are not in input_props_user
    missing_features = [prop for prop in input_props_original if prop not in input_props_user]

    if missing_features:
        print(f'Missing features: {missing_features}. Generating "fake" input features...')

        # Load input features modeled distribution
        input_transform = torch.load(f'input_features_sampler/input_features_dist.pt', map_location=device)
        dist_base = dist.Normal(torch.zeros(5, device=device), torch.ones(5, device=device) * 0.2)
        dist_input = dist.TransformedDistribution(dist_base, [input_transform])
        # Sample values
        fake_input_data = dist_input.sample(N).numpy()

        # Revert scaling
        x_scaler = joblib.load(f"input_features_sampler/input_features_scaler.save")
        fake_input_data = x_scaler.inverse_transform(fake_input_data)
        fake_input_data = pd.DataFrame(fake_input_data, columns=input_props_original)

        input_dataset.loc[:, missing_features] = fake_input_data.loc[:, missing_features]

        return input_dataset[input_props_original].to_numpy()

    else:
        print("All elements are in the other list.")
        return input_dataset[input_props_original]