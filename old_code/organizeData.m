function output_file_reference = organizeData(fileName, identifier)
    load(fileName, 'cellData', 'ReachS'); % Load the necessary variables from the file

    exportPrefex = "";

    allNeurons = 0; 
    % pull necessory structs 
    cellType = {cellData.cellID};
    timeMatrix = cellData(1).Bin10(:,1);

    % create binary matrix of fireing activity and cell type
    neuronType = [];
    binaryMatrix = [timeMatrix];
    complex_spikes = [];
    sumP = 0;
    for k=1:length(cellType)
        % seperate out PC from other cells
        if allNeurons || (~isempty(cellType{:,k}) && cellType{:,k} == "PC") 
            sumP = sumP + 1;
            % binaryMatrix = [binaryMatrix, lowpass(cellData(k).Bin10smooth(:,2), 0.25, 1/0.01)];
            binaryMatrix = [binaryMatrix, cellData(k).Bin10(:,2)];
            % if ~isempty(cellData(k).CS_Bin1)
            %     min_length = min(length(timeMatrix), length(cellData(k).CS_Bin1(:,2)));
            %     complex_spikes = [complex_spikes, cellData(k).CS_Bin1(1:min_length, 2)];
            % else
            %     complex_spikes = [complex_spikes,[]];
            % end

            neuronType = [neuronType, ~isempty(cellType{:,k}) && cellType{:,k} == "PC"];
        end
    end

    disp("total PC: " + sumP);

    scale_01_kin = {1,1,1,1,0,0,0};

    % aggrogate all data together
    simStruct = {ReachS.stim};
    kinStruct = {ReachS.filt_kin};

    kinAggrogate = struct();
    for i = 1:length(simStruct)

        % get kin data
        kinActivity = kinStruct{:,i};

        % select reach
        % threshHold_idx = find(kinActivity(:,2) > 1,1);
        % kinActivity = kinActivity(threshHold_idx-600:threshHold_idx+600,:);

        % aggorgate 
        disp("aggorgating row: " + i)
        num_columns = size(kinActivity, 2);

        % get all time in neural activations associated with kin time
        timeMask = timeMatrix > kinActivity(1,1) & timeMatrix < kinActivity(end,1);
        stimTime_0 = timeMatrix(timeMask);
        kinInterp = [stimTime_0];

        % interpolate kinematic data to fill in space for binary
        for t = 2:num_columns
            interp = interp1(kinActivity(:,1), kinActivity(:,t), kinInterp(:,1));

            if scale_01_kin{t-1} == 1
                interp = rescale(interp);
            else
                interp = rescale(interp, -1,1); 
            end

            kinInterp = [kinInterp, interp];
        end

        if ~isempty(simStruct{:,i})
            % save kinematic and neural data together
            kinAggrogate.("stim_"+simStruct{:,i}).("l_"+i) = kinInterp;
            kinAggrogate.("stim_"+simStruct{:,i}).("n_"+i) = binaryMatrix(timeMask,:);
            kinAggrogate.("stim_"+simStruct{:,i}).("label_"+i) = neuronType;

            if ~isempty(complex_spikes)
                kinAggrogate.("stim_"+simStruct{:,i}).("comp_"+i) = complex_spikes(timeMask,:);
            end
        end
    end

    % save file 
    output_file_reference = identifier+"_bin10_spike"+exportPrefex;

    save(output_file_reference, "kinAggrogate");
    disp(output_file_reference);
end