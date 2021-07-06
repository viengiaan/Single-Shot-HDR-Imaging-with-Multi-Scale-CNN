clc;clear

%% Load Images
% Non-calibrated
path = cell(12,1);
path{1} = 'Test_imgs/NonCalibrated/gt/dani_belgium.hdr';
path{2} = 'Test_imgs/NonCalibrated/gt/office.hdr';
path{3} = 'Test_imgs/NonCalibrated/gt/mpi_office.hdr';
path{4} = 'Test_imgs/NonCalibrated/gt/C08_HDR.hdr';
path{5} = 'Test_imgs/NonCalibrated/gt/C11_HDR.hdr';
path{6} = 'Test_imgs/NonCalibrated/gt/C15_HDR.hdr';
path{7} = 'Test_imgs/NonCalibrated/gt/C23_HDR.hdr';
path{8} = 'Test_imgs/NonCalibrated/gt/C25_HDR.hdr';
path{9} = 'Test_imgs/NonCalibrated/gt/C26_HDR.hdr';
path{10} = 'Test_imgs/NonCalibrated/gt/C31_HDR.hdr';
path{11} = 'Test_imgs/NonCalibrated/gt/C32_HDR.hdr';
path{12} = 'Test_imgs/NonCalibrated/gt/C42_HDR.hdr';

path_input = cell(12,1);
path_input{1} = 'Test_imgs/NonCalibrated/input/belgium.mat';
path_input{2} = 'Test_imgs/NonCalibrated/input/office.mat';
path_input{3} = 'Test_imgs/NonCalibrated/input/mpi_office.mat';
path_input{4} = 'Test_imgs/NonCalibrated/input/C08.mat';
path_input{5} = 'Test_imgs/NonCalibrated/input/C11.mat';
path_input{6} = 'Test_imgs/NonCalibrated/input/C15.mat';
path_input{7} = 'Test_imgs/NonCalibrated/input/C23.mat';
path_input{8} = 'Test_imgs/NonCalibrated/input/C25.mat';
path_input{9} = 'Test_imgs/NonCalibrated/input/C26.mat';
path_input{10} = 'Test_imgs/NonCalibrated/input/C31.mat';
path_input{11} = 'Test_imgs/NonCalibrated/input/C32.mat';
path_input{12} = 'Test_imgs/NonCalibrated/input/C42.mat';

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

%% Set up Caffe settings
% caffe.set_mode_cpu(); % Use CPU

caffe.set_mode_gpu(); % Use GPU
gpu_id = 1; % Use GPU
caffe.set_device(gpu_id); % Use GPU

%% Set up HDR-VDP settings
warning('off', 'all');
ppd = hdrvdp_pix_per_deg(21, [1024 768], 0.5);

%%  Evaluate models with HDR IQA metrices
IQA = zeros(size(path, 1), 4); % HDR-VDP: Q, P - pu_PSNR - log_PSNR

for i = 1 : size(path, 1)
	fprintf('%d image in processing.\n',i);
        hdr = hdrread(path{i});
        hdr(hdr < 0) = 0;
        
	[h, w, ~] = size(hdr);
	if mod(h, patch_size) ~= 0
		ratio = floor(h / patch_size);
		new_h = patch_size * ratio;
		hdr = hdr(1 : new_h, :, :);
	end

	if mod(w, patch_size) ~= 0
		ratio = floor(w / patch_size);
		new_w = patch_size * ratio;
		hdr = hdr(:, 1 : new_w, :);
	end

	E = hdr;

	if i == 1
		%E = hdr(1 : 768, 1 : 1024, :); % dani_belgium
		NewMax = 30000;
	end

	if i == 2
		%E = hdr(:, 1 : 1984, :); % office
		NewMax = 30000;
	end

	if i == 3
		%E = hdr(1 : 32 * 21, :, :); % mpi_office
		NewMax = 30000;
	end

	if i > 3
		%E = hdr(1 : 1056, :, :); % C images
		NewMax = 15000;
	end

	% GROUND-TRUTH
    Max = max(max(max(E)));
    N = E ./ Max;
    N = N * NewMax;
    
    % Proposed CNN
    load(path_input{i}); % E_hat
    
    model_dir = 'WEIGHTS/';
    net_weights = [model_dir 'Model_iter_150000.caffemodel'];
    net_model = 'HDRNet_mat.prototxt';
    
    HDR = CNN(E_hat, 32, 16, net_weights, net_model, 1);
    
    % HDR IQAs
    output = HDR;
    gt = N;
    
    [m, n, ~] = size(gt);
    
    % HDR-VDP
    vdp = hdrvdp(output, gt, 'rgb-bt.709', ppd, { 'surround_l', mean(mean(mean(gt)))});
    pvalue = sum(sum(vdp.P_map)) / (m * n);
    
    % PU psnr
    pu_psnr = qm_pu2_psnr(output, gt);
    
    % LOG psnr
    log_psnr = qm_log_psnr_rgb(output, gt);
    
    % Write values: HDR-VDP: Q, P - pu_PSNR - log_PSNR
    IQA(i, 1) = vdp.Q;
    IQA(i, 2) = pvalue;
    IQA(i, 3) = pu_psnr;
    IQA(i, 4) = log_psnr;
           
end

avg_Qvalue = mean(IQA(:, 1))
avg_Pvalue = mean(IQA(:, 2))

avg_PU = mean(IQA(:, 3))
avg_LOG = mean(IQA(:, 4))




















