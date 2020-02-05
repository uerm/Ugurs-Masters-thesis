function y3 = dwt_denoise3(sig3,n,data)
y3 = zeros(0); % Preallocating the y2 variable.

D = length(data);


for k = 1:D
    signal = sig3{1,k};
    
    wt = modwt(signal,n);
    wtrec = zeros(size(wt));
    wtrec(2:n-1,:) = wt(2:n-1,:);
    
    y3{1,k} = imodwt(wtrec,'db4')*(-1);
end


end