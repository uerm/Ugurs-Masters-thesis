function tensor = TensorConstruct(signal1,signal2,segment)

% Lead I
ecg_segments1 = {};
for p = 1:length(signal1)
    patient1 = signal1{1,p};
    peaks1 = segment{1,p};
    seg_lengths1 = peaks1(:,end)-peaks1(:,1)+1;
    max_len1 = max(seg_lengths1);
    patient_segments1 = zeros(0,max_len1);
    for i = 1:length(peaks1(:,1))
        seg1 = patient1(peaks1(i,1):peaks1(i,end));
        pad1 = zeros(1,max_len1-length(seg1));
        padded_seg1 = [seg1,pad1];
        patient_segments1(end+1,:) = padded_seg1;
    end
    ecg_segments1{end+1} = patient_segments1;
end

% Lead II
ecg_segments2 = {};
for p = 1:length(signal2)
    patient2 = signal2{1,p};
    peaks2 = segment{1,p};
    seg_lengths2 = peaks2(:,end)-peaks2(:,1)+1;
    max_len2 = max(seg_lengths2);
    patient_segments2 = zeros(0,max_len2);
    for i = 1:length(peaks2(:,1))
        seg2 = patient2(peaks2(i,1):peaks2(i,end));
        pad2 = zeros(1,max_len2-length(seg2));
        padded_seg2 = [seg2,pad2];
        patient_segments2(end+1,:) = padded_seg2;
    end
    ecg_segments2{end+1} = patient_segments2;
end

for i = 1:length(signal1)
    tensor{1,i} = cat(3,ecg_segments1{1,i},ecg_segments2{1,i});
end