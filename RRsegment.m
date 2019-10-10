% This segments the R peaks into segments of 128 R peaks with 50 % overlap

QRS_Length=length(QRS);

for i=1:QRS_Length
    L=1;
    K=1;
    for j=1:64:length(QRS{i})
        if (L+127) > length(QRS{i})
            continue;
        end
        Data_128{i}(K,:)=QRS{i}(L:L+127);
        L=L+64;
        K=K+1;
    end
end