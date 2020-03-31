### Code for the ECCV 2018 Paper
Fast Light Field Reconstruction With Deep Coarse-To-Fine Modeling of Spatial-Angular Clues

### ------------------------------------------------------------------------------------------
### Please also read our TIP 2018 paper: "Light Field Spatial Super-resolution Using Deep Efficient Spatial-Angular Separable Convolution" with code below
### Pytorch - https://github.com/jingjin25/LFSSR-SAS-PyTorch
### Matlab - https://github.com/spatialsr/DeepLightFieldSSR
### ------------------------------------------------------------------------------------------

### Description

A learning based model that generate a densely-sampled LF fast and accurately from a sparsely-sampled LF in one forward pass.

### Requirements and Dependencies

- MATLAB
- cuda and cudnn (For GPU. Please modify install.m if not using cudnn)
- matconvnet (Please use the matconvnet code given in this repository. It contains the 4D convolution code written by us)

### Installation

    # Start MATLAB
    $ matlab
    >> install

### Training

Set the training and validation data directory (opts.test_dir) in init_opts.m. Download the training and validation datasets to the specofoc directories. Make sure that there are enough memory for loading the whole training and validatoin datasets.

    >> train

### Testing Pretrained Models

Set the testing data directory (opts.test_dir) in init_opts.m

    >> test

### Testing Your Own Models

    >> test_model(name, depth, gpu, saveImg, epoch, len)
    
- model_name    : model name
- depth         : model depth
- gpu           : GPU ID
- saveImg       : Save the HR SAIs if true
- epoch         : model epoch to test
- len           : controls the size of the sub-lightfield, value depends on GPU memory
   
### Authors of the Paper

Henry W. F. Yeung*, Junhui Hou*, Jie Chen , Yuk Ying Chung and Xiaoming Chen

\* Equal Contibutions
