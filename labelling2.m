function AL = labelling2(anntype_new,Data)

for i = 1:length(Data)
    
    Label = zeros(0);
    
    for m = 1:length(anntype_new{1,i})            
            % AF
        if isequal(anntype_new{1,i}{m},'(AFIB') % AF
            Label(m) = 1;
        else
            Label(m) = 0;

        end
    end
    AL{i,1} = Label';
end
AL = AL';
end