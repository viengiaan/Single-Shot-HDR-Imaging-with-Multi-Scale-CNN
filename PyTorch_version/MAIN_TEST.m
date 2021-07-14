clc;clear

is_non_calibrated = 1;

path = cell(12,1);
path{1} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/dani_belgium.hdr';
path{2} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/office.hdr';
path{3} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/mpi_office.hdr';
path{4} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C08_HDR.hdr';
path{5} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C11_HDR.hdr';
path{6} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C15_HDR.hdr';
path{7} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C23_HDR.hdr';
path{8} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C25_HDR.hdr';
path{9} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C26_HDR.hdr';
path{10} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C31_HDR.hdr';
path{11} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C32_HDR.hdr';
path{12} = '/media/vgan/49D4672C2AF1B750/IMPORTANT_FOLDERS/Single-Shot_HDR/Conventional_Algorithms/DATA/Testing/NonCalibrated/C42_HDR.hdr';

image_names = cell(12, 1);
image_names{1} = 'belgium.mat';
image_names{2} = 'office.mat';
image_names{3} = 'mpi_office.mat';
image_names{4} = 'C08.mat';
image_names{5} = 'C11.mat';
image_names{6} = 'C15.mat';
image_names{7} = 'C23.mat';
image_names{8} = 'C25.mat';
image_names{9} = 'C26.mat';
image_names{10} = 'C31.mat';
image_names{11} = 'C32.mat';
image_names{12} = 'C42.mat';

patch_size = 32;

%% Proposed Algorithm
folder = 'Previous_Rad'; % USE THIS FOR COMPARISIONS (net_93.pth)

tp = strcat('test_data/Results/', folder);
sp_diff = strcat('test_data/Evaluation/Differences_map/', folder, '/');
sp_prob = strcat('test_data/Evaluation/Prob_map/', folder, '/');
sp_tone = strcat('test_data/Evaluation/Tone_mapped/', folder, '/');
sp_IQA = strcat('test_data/Evaluation/IQA/', folder, '/');

IQA = Evaluate_with_IQAs(path, image_names, is_non_calibrated, patch_size, tp, sp_diff, sp_prob, sp_tone, sp_IQA);


