- parameter_space.py: contains the python class ParameterSpace to define a python object for the
  set of features to be predicted, i.e., the dimensions of the parameter space of the modeled (joint)
  distribution.
- get_prediction.py: example script to load the models, generate predicted samples, and compare with
  the reference.

Machine learning models files:
- NNgauss.py: contains the python class for NNgauss (MLP with Gaussian Likelihood).
- NNclass.py: contains the python class for NNclass (MLP classifier + HiVAl).
- NF.py: contains the python class for Normalizing Flows.

Each of these classes contain methods to:
- Load the model's weights (get_model)
- Pre-process input data
- Return samples of the predicted distributions given some input data (get_sample)
The model's weights and scaler functions (for data pre-processing) are stored on the corresponding
directory model_name/name_of_event_space/ (e.g., NF/smass_color_sSFR_radius/).

HiVAl files:
- hival.py: contains the HiVAl python class and the functions to run HiVAl.
- run_hival.py: script to run HiVAl over a dataset.

HiVAl files are stored at:
HiVAL Drive: https://drive.google.com/drive/folders/1GYOXChEAzbyNW5Pr2CNRadGVn8WksPtz?usp=drive_link 
