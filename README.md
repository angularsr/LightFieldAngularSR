### LightFieldAngularSR
Fast Light Field Reconstruction With Deep Coarse-To-Fine Modeling of Spatial-Angular Clues
ECCV 2018

### Introduction

A learning based model that generate a densely-sampled LF fast and accurately from a sparsely-sampled LF in one forward pass.

### Requirements and Dependencies

- Matlab
- cuda and cudnn (For GPU. Please modify install.m if not using cudnn)

### Installation

    # Start MATLAB
    $ matlab
    >> install

### Training

Set the training and validation data directory (opts.test_dir) in init_opts.m. Make sure that there are enough RAM for loading the whole training and validatoin dataset.

    >> train

### Testing Pretrained Models

Set the testing data directory (opts.test_dir) in init_opts.m

    >> test


1 if using GPU, otherwise 0
Set saveImg to anything will also save the image, omitting this variable will only save the csv result table.

### Testing Your Own Models

    >> test_model(name, depth, gpu, saveImg, epoch, len)
    
- model_name    : model name
- depth         : model depth
- gpu           : GPU ID
- saveImg       : Save the SR subviews if true
- epoch         : model epoch to test
- len           : controls the size of the sub-lightfield, value depends on GPU memory
   
### Authors of the Paper

Henry W. F. Yeung*, Junhui Hou*, Jie Chen , Yuk Ying Chung and Xiaoming Chen

* Equal Contibutions
