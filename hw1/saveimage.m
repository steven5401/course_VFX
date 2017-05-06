function saveimage(gr, gg, gb, filename, numberOfImages)
value = imread(strcat(filename,'.jpg'));
outvalue = zeros(size(value), 'single');
outvalue2 = zeros(size(value), 'single');
%{
info = imfinfo(strcat(filename,'.jpg'));
lnDeltaT = log(info.DigitalCamera.ExposureTime);%
for i = 1:size(value, 1)
    for j = 1:size(value, 2)
        outvalue2(i, j, 1) = exp(gr(value(i, j, 1)+1) - lnDeltaT);
        outvalue2(i, j, 2) = exp(gg(value(i, j, 2)+1) - lnDeltaT);
        outvalue2(i, j, 3) = exp(gb(value(i, j, 3)+1) - lnDeltaT);
    end
end
%}
for k = 1:numberOfImages
    value = imread(strcat('0', num2str((str2num(filename)+k-1)),'.jpg'));
    info = imfinfo(strcat('0', num2str((str2num(filename)+k-1)),'.jpg'));
    lnDeltaT = log(info.DigitalCamera.ExposureTime);%
    for i = 1:size(value, 1)
        for j = 1:size(value, 2)
            outvalue(i, j, 1) = outvalue(i, j, 1)+exp(gr(value(i, j, 1)+1) - lnDeltaT);
            outvalue(i, j, 2) = outvalue(i, j, 2)+exp(gg(value(i, j, 2)+1) - lnDeltaT);
            outvalue(i, j, 3) = outvalue(i, j, 3)+exp(gb(value(i, j, 3)+1) - lnDeltaT);
        end
    end
end
outvalue = outvalue./numberOfImages;

hdrwrite(outvalue, strcat('o', filename, '.hdr'));

end

