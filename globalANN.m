function global_ann_sig = globalANN(sig1,sig2,sig3,ann1,ann2,ann3,...
    comments1,comments2,comments3,Data)

global_ann_sig = zeros(0);

for j = 1:length(Data)
    signal_length1 = length(sig1{j});
    signal_length2 = length(sig2{j});
    signal_length3 = length(sig3{j});
    anno_sig_1 = zeros(1,signal_length1);
    anno_sig_2 = zeros(1,signal_length2);
    anno_sig_3 = zeros(1,signal_length3);
    
    ann1_ = [ann1{j}; signal_length1];
    ann2_ = [ann2{j}; signal_length2];
    ann3_ = [ann3{j}; signal_length3];
    
    for i=1:length(comments1{j})
        if strcmp(comments1{j}{i},'(AFIB')
            anno_sig_1(ann1_(i):ann1_(i+1)) = 1;
        end
    end
    for i=1:length(comments2{j})
        if strcmp(comments2{j}{i},'(AFIB')
            anno_sig_2(ann2_(i):ann2_(i+1)) = 1;
        end
    end
    for i=1:length(comments3{j})
        if strcmp(comments3{j}{i},'(AFIB')
            anno_sig_3(ann3_(i):ann3_(i+1)) = 1;
        end
    end
    
    global_ann_sig{j} = or(or(anno_sig_1,anno_sig_2),anno_sig_3);
    
    %global_ann_sig{j} = double(global_ann_sig{j});
end