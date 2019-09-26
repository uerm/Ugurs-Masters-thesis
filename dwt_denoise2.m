function y2 = dwt_denoise2(sig2)
y2 = zeros(0);


for k = 1:48
    signal = sig2{1,k};
    
    wt = modwt(signal,5);
    wtrec = zeros(size(wt));
    wtrec(3:5,:) = wt(3:5,:);
    
    y2{1,k} = imodwt(wtrec,'db4');
end



end