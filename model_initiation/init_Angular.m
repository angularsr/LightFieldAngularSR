function net = init_Angular(opts)
% -------------------------------------------------------------------------
%   Description:
%       create initial LapSRN model
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - opts  : options generated from init_opts()
%
%   Output:
%       - net   : dagnn model
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

    %% parameters
    rng('default');
    rng(0) ;
    
    % filter width
    f       = opts.conv_f;
    % number of filters set to 64
    n       = opts.conv_n;
    % angular resolution should be 64 

    pad     = floor(f/2);
    depth   = opts.depth;
    patch_size = opts.patch_size;

    level = 1;
    a = 4;

    
    net = dagnn.DagNN;
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Feature extraction branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sigma   = opts.init_sigma;
    filters = sigma * randn(f, f, f, f, 1, n, 'single');
    biases  = zeros(1, n, 'single');
    s = 1;

	% conv: 6D Filter
	inputs  = { 'LR' };
	outputs = { 'pre_conv6d', };
	params  = { 'pre_conv6d_f', ...
		    'pre_conv6d_b'};

	net.addLayer(outputs{1}, ...
		 dagnn.Conv6D('size', size(filters), ...
			    'pad', pad, ...
			    'padAngular', pad, ...
			    'stride', 1, ...
			    'strideAngular', 1), ...
		 inputs, outputs, params);

	idx = net.getParamIndex(params{1});
	net.params(idx).value         = filters;
	net.params(idx).learningRate  = 1;
	net.params(idx).weightDecay   = 1;

	idx = net.getParamIndex(params{2});
	net.params(idx).value         = biases;
	net.params(idx).learningRate  = 0.1;
	net.params(idx).weightDecay   = 1;

    % ReLU
    inputs  = { 'pre_conv6d' };
    outputs = { 'pre_relu' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0.2), ...
                 inputs, outputs);
             
    next_input = outputs{1};
    
    %% deep conv layers (f x f x 1 x n)    
    sigma   = sqrt( 2 / (f * f * f * f* n) );
    
    for s = 1:depth
        
        
        filters = sigma * randn(f, f, 1, 1, n, n, 'single');
        biases  = zeros(1, n, 'single');

        % conv: 6D Filter
        inputs  = { next_input };
        outputs = { sprintf('conv6d_spatial_%d', s) };
        params  = { sprintf('conv6d_f_spatial_%d', s), ...
                    sprintf('conv6d_b_spatial_%d', s)};

        net.addLayer(outputs{1}, ...
                 dagnn.Conv6D('size', size(filters), ...
                            'pad', 1, ...
                            'padAngular', 0, ...
                            'stride', 1, ...
                            'strideAngular', 1), ...
                 inputs, outputs, params);

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

        % ReLU
        inputs  = { sprintf('conv6d_spatial_%d', s) };
        outputs = { sprintf('relu_spatial_%d', s) };

        net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0.2), ...
                 inputs, outputs);

        sigma   = sqrt( 2 / (f * f * 1 * 1* n) );
        filters =  sigma * randn(1, 1, f, f, n, n, 'single');
        disp(size(filters));

        % conv: 6D Filter
        inputs  = { sprintf('relu_spatial_%d', s) };
        outputs = { sprintf('conv6d_angular_%d', s) };
        params  = { sprintf('conv6d_f_angular_%d', s), ...
                    sprintf('conv6d_b_angular_%d', s) };

        net.addLayer(outputs{1}, ...
                 dagnn.Conv6D('size', size(filters), ...
                            'pad', 0, ...
                            'padAngular', 1, ...
                            'stride', 1, ...
                            'strideAngular', 1), ...
                 inputs, outputs, params);

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

        % ReLU
        inputs  = { sprintf('conv6d_angular_%d', s) };
        outputs = { sprintf('relu_angular_%d', s) };

        net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0.2), ...
                 inputs, outputs);
             
        next_input = outputs{1};
    end             
                      
    %% residual prediction layer (f x f x n x 1)
    sigma   = sqrt(2 / (f * f * 1 * 1 * n));
    filters = sigma * randn(f, f, 2, 2, n, 60, 'single');
    biases  = zeros(1, 60, 'single');

    % conv: 6D Filter
    inputs  = { next_input };
    outputs = { 'conv6d' };
    params  = { 'conv6d_f', ...
    	    'conv6d_b'};

    net.addLayer(outputs{1}, ...
	 dagnn.Conv6D('size', size(filters), ...
	            'pad', pad, ...
	            'padAngular', 0, ...
	            'stride', 1, ...
	            'strideAngular', 1), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    %% concat
    inputs  = { 'conv6d', 'LR' };
    outputs = { 'concat' };
    net.addLayer(outputs{1}, dagnn.Concat_Reshape(), inputs, outputs)

    sigma   = sqrt( 2 / (f * f * 2 * 2* n) );
    filters = sigma * randn(f, f, 2, 2, 1, 16, 'single');
    biases  = zeros(1, 16, 'single');

    % conv: 6D Filter
    inputs  = { 'concat' };
    outputs = { 'pre_conv6d_2' };
    params  = { 'pre_conv6d_f_2', ...
	    'pre_conv6d_b_2'};

    net.addLayer(outputs{1}, ...
    	 dagnn.Conv6D('size', size(filters), ...
    		    'pad', pad, ...
		    'padAngular', 0, ...
		    'stride', 1, ...
		    'strideAngular', 2), ...
	 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;

    % ReLU
    inputs  = { 'pre_conv6d_2' };
    outputs = { 'pre_relu_2' };

    net.addLayer(outputs{1}, ...
         dagnn.ReLU('leak', 0.2), ...
         inputs, outputs);

    next_input = outputs{1};

    sigma   = sqrt( 2 / (f * f * 2 * 2 * 16) );
    filters = sigma * randn(f, f, 2, 2, 16, 64, 'single');
    disp(size(filters));
    biases  = zeros(1, 64, 'single');

    % conv: 6D Filter
    inputs  = { next_input };
    outputs = { 'conv6d_post' };
    params  = { 'conv6d_f_post', ...
            'conv6d_b_post'};

    net.addLayer(outputs{1}, ...
         dagnn.Conv6D('size', size(filters), ...
                    'pad', 1, ...
                    'padAngular', 0, ...
                    'stride', 1, ...
                    'strideAngular', 2), ...
         inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;

    % ReLU
    inputs  = { 'conv6d_post' };
    outputs = { 'relu_post' };

    net.addLayer(outputs{1}, ...
         dagnn.ReLU('leak', 0.2), ...
         inputs, outputs);

    sigma   = sqrt( 2 / (f * f * 2 * 2* 64) );
    filters =  sigma * randn(f, f, 2, 2, 64, 60, 'single');
    biases  = zeros(1, 60, 'single');

    % conv: 6D Filter
    inputs  = { 'relu_post' };
    outputs = { 'conv6d_post2' };
    params  = { 'conv6d_f_post2', ...
            'conv6d_b_post2'};

    net.addLayer(outputs{1}, ...
         dagnn.Conv6D('size', size(filters), ...
                    'pad', 1, ...
                    'padAngular', 0, ...
                    'stride', 1, ...
                    'strideAngular', 1), ...
         inputs, outputs, params);

    idx = net.getParamIndex(params{1}); 
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;

    next_input = outputs{1};

    for s = level : -1 : 1

        % add
        inputs  = { next_input, ...
                    'conv6d'};
        outputs = { 'output' };
        net.addLayer(outputs{1}, ...
            dagnn.Sum(), ...
            inputs, outputs);
        
        next_input = outputs{1};
        
        %% Loss layer
        inputs  = { next_input, ...
                    'HR' };
        outputs = { sprintf('%s_loss', opts.loss) };
        
        net.addLayer(outputs{1}, ...
                 dagnn.vllab_dag_loss(...
                    'loss_type', opts.loss), ...
                 inputs, outputs);
             
                
    end   
             

end
