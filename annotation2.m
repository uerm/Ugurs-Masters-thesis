function anntype_new = annotation2(Data,loc,wave,ann,anntype)

for i = 1:length(Data)
    N{1,i} = find(loc{1,i}=='N');
    
    for no_N = 1:length(N{1,i})
        loc_N = N{1,i}(no_N);
        wave_new{1,i}(no_N,1) = wave{1,i}(loc_N);
        
        anno = ann{1,i};
        n = wave{1,i}(loc_N);
        [value,index] = min(abs(anno-n));
        QRS2{1,i}(no_N,1) = anno(index); % New QRS
        
        
        for k = 1:length(anntype{1,i})
            if anno(index) == ann{1,i}(k)
                anntype_new{1,i}(no_N,1) = anntype{1,i}(k);
            end
        end
    end
end

end