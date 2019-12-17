function M = threshold(AL_128,n)

for i = 1:length(AL_128)
    for k = 1:size(AL_128{1,i},1)
        M{i}(k,:) = double(mean(AL_128{1,i}(k,:))>=(n/100));
    end
end