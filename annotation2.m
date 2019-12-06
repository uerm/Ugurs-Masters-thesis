function anntype_new = annotation2(Data,loc,wave,ann,anntype)

for i = 1:length(Data)
    N_loc_matrix{1,i} = find(loc{1,i}=='N');
    
    for N_no = 1:length(N_loc_matrix{1,i})
        N_loc = N_loc_matrix{1,i}(N_no);
        wave_new{1,i}(N_no,1) = wave{1,i}(N_loc);
        
        a = ann{1,i};
        n = wave{1,i}(N_loc);
        [val,idx] = min(abs(a-n));
        QRS2{1,i}(N_no,1) = a(idx); % New QRS
        
        
        for j = 1:length(anntype{1,i})
            if a(idx) == ann{1,i}(j)
                anntype_new{1,i}(N_no,1) = anntype{1,i}(j);
            end
        end
    end
end

end

%anntype_new = segmentation(anntype_new);

%%

% Label = zeros(0);
% 
% for m = 1:length(anntype_new{1,i})
%     % Normal
%     if double(anntype_new{1,i}(m)) == 78 % N
%         Label(m) = 1;
%     elseif double(anntype_new{1,i}(m)) == 76 % L
%         Label(m) = 1;
%     elseif double(anntype_new{1,i}(m)) == 82 %R
%         Label(m) = 1;
%     elseif double(anntype_new{1,i}(m)) == 101 %e
%         Label(m) = 1;
%     elseif double(anntype_new{1,i}(m)) == 106 %j
%         Label(m) = 1;
%         
%     % AF    
%     elseif double(anntype_new{1,i}(m)) == 83 % S
%         Label(m) = 2;
%     elseif double(anntype_new{1,i}(m)) == 65 % A
%         Label(m) = 2;
%     elseif double(anntype_new{1,i}(m)) == 97 % a
%         Label(m) = 2;
%     elseif double(anntype_new{1,i}(m)) == 74 % J
%         Label(m) = 2;
%     
%         
%     % Other    
%     else
%         Label(m) = 3;
%     end
% end
       
    

