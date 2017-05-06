function [Zr,Zg,Zb,B,w ] = parseimage(filename, pixelPositionH, pixelPositionW, numberOfImages)
B = zeros(numberOfImages,1);
%B=[1/750, 1/180, 1/45];
Zr = zeros(size(pixelPositionH, 2), numberOfImages);
Zg = zeros(size(pixelPositionH, 2), numberOfImages);
Zb = zeros(size(pixelPositionH, 2), numberOfImages);
w = zeros(256,1);
for j = 1:numberOfImages
    info = imfinfo(strcat('0', num2str((str2num(filename)+j-1)),'.jpg'));
    value = imread(strcat('0', num2str((str2num(filename)+j-1)),'.jpg'));
    B(j) = log(info.DigitalCamera.ExposureTime);
    for i = 1:size(pixelPositionH, 2)
        Zr(i, j) = value(pixelPositionH(i), pixelPositionW(i), 1);
        Zg(i, j) = value(pixelPositionH(i), pixelPositionW(i), 2);
        Zb(i, j) = value(pixelPositionH(i), pixelPositionW(i), 3);
    end
end
for i = 1:256
    if i > 128
        w(i) = 256-i;
    else
        w(i) = i-1;
    end
end
end


