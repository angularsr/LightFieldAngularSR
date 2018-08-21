function [fullLF] = ReadIllumImages(scenePath)

numImgsX = 14;
numImgsY = 14;

%%% converting the extracted light field to a different format
inputImg = im2double(imread(scenePath));

h = size(inputImg, 1) / numImgsY;
w = size(inputImg, 2) / numImgsX;
fullLF = zeros([h, w, numImgsY, numImgsX]);
for ax = 1 : numImgsX
    for ay = 1 : numImgsY
        img = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
        img = rgb2ycbcr(img);
        fullLF(:, :, ay, ax) = im2single(img(:, :, 1));
    end
end

fullLF = fullLF(:, :, 4:11, 4:11); % we only take the 8 middle images
[f_w, f_h, f_n1, f_n2] = size(fullLF);
fullLF = reshape(fullLF, [f_w, f_h, f_n1*f_n2]);
