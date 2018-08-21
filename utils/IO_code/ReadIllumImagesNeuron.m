function [fullLF, inputLF] = ReadIllumImages(scenePath)

numImgsX = 16;
numImgsY = 16;

%%% converting the extracted light field to a different format
inputImg = im2double(imread(scenePath));

h = size(inputImg, 1) / numImgsY;
w = size(inputImg, 2) / numImgsX;

inputLF = zeros(floor(h), floor(w), 3, numImgsY, numImgsX);
fullLF = zeros(floor(h), floor(w), 3, numImgsY, numImgsX);

for ax = 1 : numImgsX
    for ay = 1 : numImgsY
        img =  inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
        fullLF(:, :, :, ay, ax) = img;
        img = rgb2ycbcr(img);
        inputLF(:, :, :, ay, ax) = img;
    end
end

fullLF = fullLF(:, :, :, 5:11, 5:11); % we only take the 8 middle images
inputLF = inputLF(:, :, :, 5:11, 5:11); % we only take the 8 middle images
[f_w, f_h, f_c, f_n1, f_n2] = size(inputLF);
inputLF = reshape(inputLF, [f_w, f_h, f_c, f_n1*f_n2]);
fullLF = reshape(fullLF, [f_w, f_h, f_c, f_n1*f_n2]);

