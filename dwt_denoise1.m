function y1 = dwt_denoise1(sig1)
y1 = zeros(0);


for k = 1:48
    signal = sig1{1,k}; % All 48 records
    
    wt = modwt(signal,9); % Maximal overlap DWT with 5 levels
    wtrec = zeros(size(wt));
    wtrec(1:8,:) = wt(1:8,:);
    
    y1{1,k} = imodwt(wtrec,'db4')*(-1); % Reconstruction with only subband 3 to 5 used.
end




end
