function y1 = dwt_denoise1(sig1,n,data)
y1 = zeros(0); % Preallocating the y1 variable.

D = length(data);


for k = 1:D
    signal = sig1{1,k}; % All 48 records
    
    wt = modwt(signal,n); % Maximal overlap DWT with n levels
    wtrec = zeros(size(wt));
    wtrec(2:n-1,:) = wt(2:n-1,:);
    
    y1{1,k} = imodwt(wtrec,'db4')*(-1); % Reconstruction with only subband 1 to n-1 used.
end


end