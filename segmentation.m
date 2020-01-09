function seg = segmentation(input,n)

nn = ones(1)*n;

seg_length=length(input);

seg = zeros(0);

for i=1:seg_length
    L=1;
    K=1;
    m=0;
    for j=1:(nn/2):length(input{i})
        if (L+nn) > length(input{i})
            continue;
        end
        seg{i}(K,:)=input{i}(((nn/2)*m+1):(L+(nn-1)));
        L=L+(nn/2);
        K=K+1;
        m=m+1;
    end
end