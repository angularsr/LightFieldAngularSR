function train_model(name, depth, gpu, imdb)
% -------------------------------------------------------------------------
%   Description:
%       Script to train the Angular SR from scratch
%	Modified from the code produced by the authors in the citation below
%
%   Input:
%       - scale : SR upsampling scale
%       - depth : numbers of conv layers in each pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%       - imdb  : the loaded dataset
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------


    %% initialize opts
    opts = init_opts(name, depth, gpu);

    %% save opts
    filename = fullfile(opts.train.expDir, 'opts.mat');
    fprintf('Save parameter %s\n', filename);
    save(filename, 'opts');

    %% setup paths
    addpath(genpath('utils/IO_code'));
    addpath(genpath('utils/training_code'));
    addpath(genpath('model_initiation'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% initialize network
    fprintf('Initialize network...\n');
    model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

    if( ~exist(model_filename, 'file') )
        model = eval(['init_' name '(opts)']);
        fprintf('Save %s\n', model_filename);
        net = model.saveobj();
        save(model_filename, 'net');
    else
        fprintf('Load %s\n', model_filename);
        model = load(model_filename);
        model = dagnn.DagNN.loadobj(model.net);
    end

    %% training
    get_batch = @(x,y,mode,nets) getBatch(opts,x,y,mode);

    %model = reinit_params(model);
    [net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                      'val', find(imdb.images.set == 2));
