function seg = segmentation(input)

seg_length=length(input);

seg = zeros(0);

for i=1:seg_length
    L=1;
    K=1;
    m=0;
    for j=1:64:length(input{i})
        if (L+128) > length(input{i})
            continue;
        end
        seg{i}(K,:)=input{i}((64*m+1):(L+127));
        L=L+64;
        K=K+1;
        m=m+1;
    end
end