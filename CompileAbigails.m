% Define the input and output folders
inputFolder = '/Users/gunnarenserro/Documents/AlBornoMazenlab/data/AbigailUpdated'; % Path to the folder containing your files
outputFolder = '/Users/gunnarenserro/Documents/AlBornoMazenlab/data/compiled_Abigail_bin10'; % Path to the folder where you want to save the output

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get a list of all .mat files in the input folder
files = dir(fullfile(inputFolder, '*.mat'));

% Loop through each file and process it
for i = 1:length(files)
    % Get the full file name
    fileName = fullfile(inputFolder, files(i).name);
    
    % Display the file being processed
    disp(['Processing file: ', fileName]);
    
    try

        % Extract identifiers
        pattern = '(?<name>\w+)_(?<date>\d{6})_(?<identifier>\w+)_NPresults';
        tokens = regexp(fileName, pattern, 'names');


        % Call the function to process the file
        output_file_name = organizeData(fileName, tokens.date + "_" + tokens.identifier);
        
        
        % Move the generated file to the output folder
        generatedFile = dir('*_*_bin10_spike.mat'); % Pattern to match generated files
        if ~isempty(generatedFile)
            movefile(generatedFile(1).name, fullfile(outputFolder, generatedFile(1).name));
        end
        
    catch ME
         % Display error message and file that caused the error
        disp(['Error processing file: ', fileName]);
        disp(['Error message: ', ME.message]);
    end
end