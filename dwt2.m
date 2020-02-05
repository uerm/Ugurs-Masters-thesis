function y2 = dwt2(sig2,n,data)
y2 = zeros(0); % Preallocating the y1 variable.

D = length(data);


for k = 1:D
    signal = sig2{1,k}; % All records
    
    wt = modwt(signal,n); % Maximal overlap DWT with n levels
    wtrec = zeros(size(wt));
    wtrec(2:n-1,:) = wt(2:n-1,:);
    
    y2{1,k} = imodwt(wtrec,'db4')*(-1); % Reconstruction with only subband 1 to n-1 used.
end


end