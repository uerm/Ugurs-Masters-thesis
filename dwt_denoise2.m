function y2 = dwt_denoise2(sig2,n,data)
y2 = zeros(0); % Preallocating the y2 variable.

D = length(data);


for k = 1:D
    signal = sig2{1,k};
    
    wt = modwt(signal,n);
    wtrec = zeros(size(wt));
    wtrec(2:n-1,:) = wt(2:n-1,:);
    
    y2{1,k} = imodwt(wtrec,'db4')*(-1);
end


end