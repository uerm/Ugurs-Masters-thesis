cnt = 1;
ecg_segments2 = {};
for p = 1:length(y2)
    patient2 = y2{1,p};
    peaks2 = Data_128{1,p};
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