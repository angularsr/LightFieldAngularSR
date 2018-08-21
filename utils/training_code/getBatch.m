function inputs = getBatch(opts, imdb, batch, mode)
% -------------------------------------------------------------------------
%   Description:
%       get one batch for training LapSRN
%       Modified from the code produced by the authors in the citation below
%
%   Input:
%       - opts  : options generated from init_opts()
%       - imdb  : imdb file generated from make_imdb()
%       - batch : array of ID to fetch
%       - mode  : 'train' or 'val'
%
%   Output:
%       - inputs: input for dagnn (include LR and HR images)
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

    %% get images
    image_batch = imdb.images.img(batch);
    
    %% crop
    HR = zeros(opts.patch_size, opts.patch_size, opts.conv_a, length(batch), 'single');
    
    for i = 1:length(batch)
        
        img = image_batch{i};
              
        H = size(img, 1);
        W = size(img, 2);
        
        % random crop
        r1 = floor(opts.patch_size / 2);
        r2 = opts.patch_size - r1 - 1;
        
        mask = zeros(H, W);
        mask(1 + r1 : end - r2, 1 + r1 : end - r2) = 1;
        
        [X, Y] = meshgrid(1:W, 1:H);
        X = X(mask == 1);
        Y = Y(mask == 1);
        
        select = randperm(length(X), 1);
        X = X(select);
        Y = Y(select);

        HR(:, :, :, i) = img(Y - r1 : Y + r2, X - r1 : X + r2, :);
    end
 
    if ( randi(2) - 1)
        HRIn = HR(:,:,flip([2:7,9:56,58:63]),:);
        HRIn = reshape(HRIn, [64,64,1,1,60]);
        HR = permute(HR, [1,2,4,3]);
        tmp = HR(:,:,:,flip([1,8,57,64]));
        tmp = reshape(tmp, [64,64,2,2,1]);

    else

        HRIn = HR(:,:,[2:7,9:56,58:63],:);
        HRIn = reshape(HRIn, [64,64,1,1,60]);
        HR = permute(HR, [1,2,4,3]);
        tmp = HR(:,:,:,[1,8,57,64]);
        tmp = reshape(tmp, [64,64,2,2,1]);

    end

    %% make dagnn input
    inputs = {};
    inputs{end+1} = 'HR';
    inputs{end+1} = HRIn;
    
    inputs{end+1} = 'LR';
    inputs{end+1} = tmp;
    
    % convert to GPU array
    if( opts.gpu > 0 )
        for i = 2:2:length(inputs)
            inputs{i} = gpuArray(inputs{i});
        end
    end
end
