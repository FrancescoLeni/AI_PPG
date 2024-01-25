# AI_PPG
Repo for Applied AI PPG classification project

The repo is structured to handle the training of a great variety of models and architectures.

* 'organize_dataset.py' is used to rearrange the data in more handy format and to extract crops.
* DL models are trained and testing using 'DL_train.py' 
* ML training can be handled using 'ML_experiments.py' 
* 'NB_ML_inference.ipynb' is used for ML models testing
* Creation of ML models can be seen inside 'NB_ML_all.ipynb'
* The 'pyPPG' folders contains a copy of the pyPPG library (copied to overcome compatibility limitations)
* The 'prepocessing' folder contains some modified scrips extracted from pyPPG
* 'utils' folder contains all the utility functions, divided for DL and ML
* 'models' folder contains the definition of the model classes for the DL models

to train DL model
```
python DL_train.py --model <ur_model_name> --epochs <num_epochs> --batch_size <batch_size>  --crops_raw --mode <binary or all>
```

to test a trained DL model
```
python DL_train.py --test --model <ur_model_weights> --epochs 1 --batch_size <batch_size>  --crops_raw --mode <binary or all>
```

