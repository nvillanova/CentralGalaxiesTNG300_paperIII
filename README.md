This repository contains the material presented in Rodrigues, de Santi, Abramo and Montero-Dorta 2024.
In this work we compare three machine learning methods (NNgauss, NNclass, NF) to model the (joint) 
distribution of central galaxy properties based on their host dark matter halo properties using the 
IllustrisTNG300 simulation.
With the available material, one can load the models (weights) and generate samples of stellar mass, 
color, sSFR, and radius, given a set of input halo properties.

**Files description:**

- get_prediction.ipynb: example script to load the models, generate samples, and compare with the reference.
- parameter_space.py: contains the python class ParameterSpace to define an object for the set of features
  to be predicted, i.e., the dimensions of the parameter space of the modeled (joint) distribution. This is
  used by HiVAl, NNgauss, NNclass and NF classes (see below).

  **Machine learning models files:**
- NNgauss.py: contains the python class for NNgauss (MLP with Gaussian Likelihood).
- NNclass.py: contains the python class for NNclass (MLP classifier + HiVAl).
- NF.py: contains the python class for Normalizing Flows.

  Each of these classes contain methods to:
- Load the model's weights (get_model).
- Pre-process input data.
- Return samples of the predicted distributions given some input data (get_sample).

  The model's weights and scaler functions (for data pre-processing) are stored on the corresponding
  directory model_name/name_of_event_space/ (e.g., NF/smass_color_sSFR_radius/).

**HiVAl files:**
- hival.py: contains the HiVAl python class and the functions to run HiVAl.
- run_hival.ipynb: script to run HiVAl over a dataset.

  HiVAl files are stored at:
  HiVAL Drive: https://drive.google.com/drive/folders/1GYOXChEAzbyNW5Pr2CNRadGVn8WksPtz?usp=drive_link 

  **Data files:**
- subtest_tng300_inputs.csv: contains the input halo properties of 10 instances randomly selected from our test set.
- subtest_tng300_targets.csv: contains the output galaxy properties of the same 10 instances randomly selected from 
  our test set.
- subtest_tng300_inputs.csv: same as subtest_tng300_inputs.csv, excluding the overdensity feature to illustrate how 
to use the models in case there are missing features (see below).


**Missing features**: in case the user does not have all halo properties computed, we make available a code to replace 
the missing feature(s) by a random sample of the distribution of the feature (learned with NF). This works similarly to 
permutation feature importance, where a feature is replaced with noise (sampled from the same distribution of the 
feature).
