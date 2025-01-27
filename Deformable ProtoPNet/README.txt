Since uploading complete folders did not work, the deformable convolution must be downloaded from: https://github.com/jdonnelly36/Deformable-ProtoPNet/tree/main.

Instructions are similar to the deformable protopnet:

Instructions for building Deformable-Convolution-V2:
1. Navigate to the Deformable-Convolution-V2-PyTorch subdirectory
2. Run make.sh

Instructions for training the model:
1. Run main.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-m is the margin to use for subtractive margin cross entropy
-last_layer_fixed is a boolean indicating whether the last layer connections will be optimized during training
-subtractive_margin is a boolean indicating whether to use subtractive margin during training or not
-using_deform is a boolean indicating whether to use subtractive margin during training or not
-topk_k is an integer indicating the number 'k' of top activations to consider; k=1 was used in our experiments
-num_prototypes is an integer indicating the number of prototypes to use (must be a multiple of the number of classes in the dataset)
-incorrect_class_connection is the value incorrect class connections are initialized to
-deformable_conv_hidden_channels is the integer number of hidden channels to use on offset prediction branch
-rand_seed is an integer setting the random seed to use for this experiment

Recommended values for all arguments on CUB_200 can be found in run.sh

Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-imgdir is the directory containing the image you want to analyze
-img is the filename of the image you want to analyze
-imgclass is the (0-based) index of the correct class of the image

Instructions for finding the nearest patches to each prototype:
1. Run global_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze


Some changed made are seperate local_analysis.py files:
1. The local_analysis_exp1_3.py is used for experiments 1 and 3, where all images of a heldout test dataset can be run in one go. Run local_analysis_exp1_3.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-testdir is the dataset containing the images you want to analyze

2. The local_analysis_exp2.py file is used for experiment 2, where all images of a heldout test dataset can be run, however, it will only provide explanations for the first 5 images of the good class for imbalanced scenarios to reduce the time of analysis. You can run this by supplying the same arguments as the .py file above:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-testdir is the dataset containing the images you want to analyze

Within main.py, changes are made to incorporate incorporate the distillation loss function.
1. To use the function, uncomment lines 201–211 and comment out lines 215–216.
2. To disable the function, uncomment lines 215–216 and comment out lines 201–211.

