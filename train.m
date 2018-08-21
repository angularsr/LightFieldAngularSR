%clear;
 
%  train_model(depth, gpu, saveImg)
%  depth:    model depth
%  gpu:      gpu_ID
%  saveImg:  save img output, 0 or 1

addpath(genpath('utils/'));

IM = loadIM;
train_model('Angular',8,1,IM);
