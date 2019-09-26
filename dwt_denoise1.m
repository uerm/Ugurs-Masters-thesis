function y1 = dwt_denoise1(sig1)
y1 = zeros(0);


for k = 1:48
    signal = sig1{1,k};
    
    wt = modwt(signal,5);
    wtrec = zeros(size(wt));
    wtrec(3:5,:) = wt(3:5,:);
    
    y1{1,k} = imodwt(wtrec,'db4');
end



end