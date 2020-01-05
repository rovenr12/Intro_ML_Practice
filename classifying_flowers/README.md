# Data Scientist Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, you will first develop code for an image classifier built with PyTorch, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.


### The use of predict.py and train.py

The train and predict.py can run under the command terminal.
Train.py should be used first in order to create the model.
Predict.py can provide the top 5 classes the image belongs to.

# Train.py

Basic use: 
  `python train.py data_directory`
  
Options:
  - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
  - Choose architecture: `python train.py data_dir --arch "vgg13"`
  - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
  - Use GPU for training: `python train.py data_dir --gpu`
  
  
It will save the model.pth and prints out training loss, validation loss, and validation accuracy as the network trains

# Predict.py

Basic usage: `python predict.py /path/to/image checkpoint`

Options:
  - Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
  - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
  - Use GPU for inference: `python predict.py input checkpoint --gpu`


it will return the top 5 flower name and class probability.
