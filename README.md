# Exploring the halo-galaxy connection with probabilistic approaches

This repository contains some material presented in Rodrigues, de Santi, Abramo and Montero-Dorta 2024.
In this work we compare three machine learning methods (NNgauss, NNclass, NF) to model the (joint) 
distribution of central galaxy properties based on their host dark matter halo properties using the 
IllustrisTNG300 simulation.
With the available material, one can load the models (weights) and generate samples of stellar mass, 
color, sSFR, and radius, given a set of input halo properties from [IllustrisTNG-300](https://www.tng-project.org) data.

## **Files description**

### **Main files**

#### **Notebooks**
- `get_predictions.ipynb`: example script to load the models, generate samples, and compare with the reference.
- `get_predictions-missing_features.ipynb`: example script to load the models, generate samples, and compare with the reference if there is some halo features missing. We stress that is is for user purposes only and do not reflect the results presented in the paper (done with the complete dataset).
- `run_hival.ipynb`: script to run `HiVAl` over a dataset of random (fake) data.

#### **Machine learning models files**
- `NNgauss.py`: contains the python class for NNgauss (MLP with Gaussian Likelihood).
- `NNclass.py`: contains the python class for NNclass (MLP classifier + HiVAl).
- `NF.py`: contains the python class for Normalizing Flows.

  Each of these classes contain methods to:
- Load the model's weights (get_model).
- Pre-process input data.
- Return samples of the predicted distributions given some input data (get_sample).

  The model's weights and scaler functions (for data pre-processing) are stored on the corresponding
  directory model_name/name_of_event_space/ (e.g., NF/smass_color_sSFR_radius/).

#### **HiVAl files**
- `hival.py:` contains the HiVAl python class and the functions to run HiVAl.

#### **Utility files**
- `parameter_space.py`: contains the python class ParameterSpace to define an object for the set of features to be predicted, i.e., the dimensions of the parameter space of the modeled (joint) distribution. This is used by HiVAl, NNgauss, NNclass and NF classes (see below).
- `missing_features.py`: in case the user does not have all halo properties computed, we make available a code to replace 
the missing feature(s) by a random sample of the distribution of the feature (learned with NF). This works similarly to 
permutation feature importance, where a feature is replaced with noise sampled from the same distribution of the 
feature. Notice that this is not equivalent to marginalizing over the missing feature.

### **Data files**

Files in the `data` folder:
- `subtest_tng300.csv`: contains the input halo and output galaxy properties of 10 instances randomly selected from our test set.
- `subtest_tng300_inputs.csv`: contains the input halo properties of 10 instances randomly selected from our test set.
- `subtest_tng300_targets.csv`: contains the output galaxy properties of the same 10 instances randomly selected from 
  our test set.
- `subtest_tng300_inputs_missing_Delta3_h.csv`: same as subtest_tng300_inputs.csv, excluding the overdensity feature to illustrate how 
to use the models in case there are missing features.

### **Results**

Files in the `results` folder:
- `linear_comparison.png`: image generated for linear predictions versus true comparisons (from `get_predictions.ipynb`)
- `from_missing-linear_comparison`: image generated for linear predictions versus true comparisons (from `get_predictions-missing_features.ipynb`)

### **Models**

Files in `models` folder: contains the trained models for `NF`, `NNclass`, and `NNgauss`.

### **Voronoi targets**

Files in `voronoi_targets` folder:
- `voronoi_targets/smass_color_sSFR_radius/`: contains the files to generate the galaxy properties samples with NNclass.
- `voronoi_targets/x_y/`: contains the output files of the example from run_hival.ipynb.

### **Input features sampler**

Files in `input_features_sampler` folder: contains the created features used in `get_predictions-missing_features.ipynb`.

## **Data download**

The complete dataset of halo and galaxy properties from `IllustrisTNG-300` can be downloaded from [https://www.tng-project.org](https://www.tng-project.org) website.
Details regarding the data pre-processing are found in the paper.

## **Contact**

For further information, suggestions or doubts get in touch with:
* Natália Rodrigues: natvnr@gmail.com or natalia.villa.rodrigues@usp.br
* Natalí de Santi: natalidesanti@gmail.com

## **Request**

If you use any of these codes or models presented here, please cite the paper.