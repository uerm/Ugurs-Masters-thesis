function AL = labelling(anntype_new,Data)

for i = 1:length(Data)
    
    Label = zeros(0);
    
    for m = 1:length(anntype_new{1,i})
        % Normal
        if double(anntype_new{1,i}(m)) == 78 % N
            Label(m) = 1;
        elseif double(anntype_new{1,i}(m)) == 76 % L
            Label(m) = 1;
        elseif double(anntype_new{1,i}(m)) == 82 %R
            Label(m) = 1;
        elseif double(anntype_new{1,i}(m)) == 101 %e
            Label(m) = 1;
        elseif double(anntype_new{1,i}(m)) == 106 %j
            Label(m) = 1;
            
            % AF
        elseif double(anntype_new{1,i}(m)) == 83 % S
            Label(m) = 2;
        elseif double(anntype_new{1,i}(m)) == 65 % A
            Label(m) = 2;
        elseif double(anntype_new{1,i}(m)) == 97 % a
            Label(m) = 2;
        elseif double(anntype_new{1,i}(m)) == 74 % J
            Label(m) = 2;
            
            
            % Other
        else
            Label(m) = 3;
        end
    end
    AL{i,1} = Label;
end
AL = AL';