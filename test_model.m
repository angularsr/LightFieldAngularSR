function test_model(name, depth, gpu, saveImg, epoch, len)
% -------------------------------------------------------------------------
%   Description:
%       Testing A Specific Model
%       Compute PSNR and SSIM
%
%   Input:
%       - model_name    : model name (Angular)
%       - depth         : model depth
%       - gpu           : GPU ID
%       - saveImg       : Save the SR subviews if true
%       - epoch         : model epoch to test
%       - len           : controls the size of the sub-lightfield, value depends on GPU memory
% -------------------------------------------------------------------------
    
    %% setup paths
    addpath(genpath('utils/IO_code'));
    addpath(genpath('utils/training_code'));
    addpath(genpath('utils/testing_code'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% generate opts
    opts = init_opts(name, depth, gpu);
    crop = 8;
    
    %% Load model
    model_filename = fullfile(opts.train.expDir, sprintf('net-epoch-%d.mat', epoch));
    fprintf('Load %s\n', model_filename);
    if(saveImg)
        mkdir(strcat(['Save_Img/' opts.model_name]))
    end
    
    net = load(model_filename);
    net = dagnn.DagNN.loadobj(net.net);
    gpuDevice(opts.gpu);
    net.move('gpu');

    %% load image list
    img_list = load_list(['lists/' opts.test_dataset '.txt']);
    num_img = length(img_list);

    %% testing
    PSNR = zeros(num_img, 1);
    SSIM = zeros(num_img, 1);
    PSNR_var = zeros(num_img, 1);
    SSIM_var = zeros(num_img, 1);
    
    for i = 1:num_img
                
        img_name = img_list{i};
        fprintf('Process Test Set %d/%d: %s\n', i, num_img, img_name);

        if(saveImg)
            mkdir(strcat(['Save_Img/' opts.model_name '/'  img_name]))
        end
    
        % Load HR image
        input_dir = opts.test_dir;
        input_filename = fullfile(input_dir, sprintf('%s.png', img_name));
        [inputLF] = ReadIllumImages(input_filename);
        [h,w,n] = size(inputLF);
        inputLF = reshape(inputLF, [h,w,1,n]);
        
        input_left = inputLF(:,1:len+crop,:,:);
        inTensor = {};
        slice_data = floor(( w - len - crop )/ len );
        for sl = 1:slice_data
            inTensor{end+1} = inputLF(:,sl*len+1-crop:(sl+1)*len+crop,:,:);
        end
        input_right = inputLF(:,end-len+1-crop:end,:,:);
        
        img_HR = zeros([h,w,60]);
        img_HR(crop+1:end-crop,crop+1:len,:) = Angular_SR(input_left, net, opts);
        for sl = 1:slice_data
            img_HR(crop+1:end-crop,1+sl*len:(sl+1)*len,:) = Angular_SR(inTensor{sl}, net, opts);
        end
        img_HR(crop+1:end-crop,end-len+1:end-crop,:) = Angular_SR(input_right, net, opts);
      
        img_HR = reshape(img_HR, [h,w,1,60]);

        %% evaluate
        psnr_score = [];
        ssim_score = [];

        inputLF = inputLF(:,:,:,[2:7,9:56,58:63]);
       
        for view = 1:60
            
            tmp_HR = img_HR(:,:,:,view);
            tmp_LR = inputLF(:,:,:,view);

           % Quantise pixels
            tmp_HR = im2double(im2uint8(tmp_HR));     

            % crop boundary
            tmp_HR = shave_bd(tmp_HR, 22);
            tmp_LR = shave_bd(tmp_LR, 22);       

            if(saveImg)
                imwrite(tmp_HR, strcat(['Save_Img/' opts.model_name '/' img_name '/' int2str(view), '.png']))
            end    

            % evaluate
            psnr_score(end+1) = psnr(tmp_HR, tmp_LR);
            ssim_score(end+1) = ssim(tmp_HR, tmp_LR);

        end
        
        % average
        PSNR(i) = mean(psnr_score);
        SSIM(i) = mean(ssim_score);
        PSNR_var(i) = var(psnr_score);
        SSIM_var(i) = var(ssim_score);
        
        disp(PSNR(i));
        disp(SSIM(i));       

    end
    
    disp('PSNR');
    disp(PSNR);
    disp('SSIM');
    disp(SSIM);
    disp('PSNR_VAR');
    disp(PSNR_var);
    disp('SSIM_VAR');
    disp(SSIM_var);
    
    PSNR_mean = mean(PSNR);
    SSIM_mean = mean(SSIM);
 
    % write result to csv file
    d = size(PSNR, 1);
    w = zeros([d,4]);
    w(:,1) = PSNR;
    w(:,2) = PSNR_var;
    w(:,3) = SSIM;
    w(:,4) = SSIM_var;
    %csvwrite(strcat(['Result_CSV/', 'Model_scale_', num2str(model_scale), '_depth_', num2str(depth),'_testscale_', num2str(test_scale), '.csv']),w)
    
    fprintf('Average PSNR = %f\n', PSNR_mean);
    fprintf('Average SSIM = %f\n', SSIM_mean);

