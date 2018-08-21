function opts = init_opts(name, depth, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Generate all options
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - scale : SR upsampling scale
%       - depth : number of conv layers in one pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%
%   Output:
%       - opts  : all options for Spatial SR
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
    if(nargin == 0)
        name = ''; depth=1; gpu=1;
    end

    %% network options
    opts.name               = name;
    opts.depth              = depth;
    opts.weight_decay       = 0.0001;
    opts.init_sigma         = 0.001;
    opts.conv_f             = 3;
    opts.conv_n             = 64;       % number of intermediate feature maps
    opts.conv_a             = 64;       % angular resolution
    opts.conv_a_l           = 4;        % input angular resolution
    opts.loss               = 'L2';

    %% training options
    opts.gpu                = gpu;
    opts.batch_size         = 1;
    opts.num_train_batch    = 1000;     % number of training batch in one epoch
    opts.num_valid_batch    = 100;      % number of validation batch in one epoch
    opts.lr                 = 1e-6;     % initial learning rate
    opts.lr_step            = 10000;    % number of epochs to drop learning rate
    opts.lr_drop            = 0;        % learning rate drop ratio
    opts.lr_min             = 1e-7;     % minimum learning rate
    opts.patch_size = 64;
    opts.data_augmentation  = 1;

    %% dataset options
    opts.train_dir              = './data/Training'; % Path to traning images
    opts.valid_dir              = './data/Validation'; % Path to validation images
    opts.test_dir               = './data/Testing'; % Path to testing images
    opts.train_dataset          = {};
    opts.train_dataset{end+1}   = 'SIG_DATA'; % File name in the list folder
    opts.valid_dataset          = {};
    opts.valid_dataset{end+1}   = 'Valid';    % File name in the list folder
    opts.test_dataset           = '30Scene';  % File name in the list folder

    %% setup model name
    opts.data_name = 'train';
    for i = 1:length(opts.train_dataset)
        opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
    end

    opts.net_name = sprintf('%s_depth%d_%s', ...
                            opts.name, opts.depth, opts.loss);

    opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s', ...
                            opts.net_name, ...
                            opts.data_name, opts.patch_size, ...
                            num2str(opts.lr), opts.lr_step, ...
                            num2str(opts.lr_drop), num2str(opts.lr_min));


    %% setup dagnn training parameters
    if( opts.gpu == 0 )
        opts.train.gpus     = [];
    else
        opts.train.gpus     = [opts.gpu];
    end
    opts.train.batchSize    = opts.batch_size;
    opts.train.numEpochs    = 10000;
    opts.train.continue     = true;
    opts.train.learningRate = learning_rate_policy(opts.lr, opts.lr_step, opts.lr_drop, ...
                                                   opts.lr_min, opts.train.numEpochs);

    opts.train.expDir = fullfile('models', opts.model_name) ; % model output dir
    if( ~exist(opts.train.expDir, 'dir') && ~strcmp(name, '') )
        mkdir(opts.train.expDir);
    end

    opts.train.model_name       = opts.model_name;
    opts.train.num_train_batch  = opts.num_train_batch;
    opts.train.num_valid_batch  = opts.num_valid_batch;
    
    % setup loss
    opts.train.derOutputs = {};
    opts.train.derOutputs{end+1} = sprintf('%s_loss',opts.loss);
    opts.train.derOutputs{end+1} = 1;

end
