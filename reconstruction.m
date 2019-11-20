function [zSampled] = reconstruction(encoderNet, x)
compressed = forward(encoderNet, x);
d = size(compressed,1)/2;
zMean = compressed(1:d,:);

sz = size(zMean);
z = zMean;
z = reshape(z, [1,1,sz]);
zSampled = dlarray(z, 'SSCB');
end