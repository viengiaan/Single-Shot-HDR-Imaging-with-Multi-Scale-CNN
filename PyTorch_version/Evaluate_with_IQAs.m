function [IQA, IQA_raw] = Evaluate_with_IQAs(path, image_names, is_non_calibrated, patch_size, tp, sp_diff, sp_prob, sp_tone, sp_IQA)

    % tp: test path
    % sp_diff: save path for differences map
    % sp_prob: ......... for HDR-VDP P map
    % sp_tone: ......... for tone mapped images
    % sp_IQA: .......... for IQA results

    warning('off', 'all');
    ppd = hdrvdp_pix_per_deg(21, [1024 768], 0.5);

    IQA = zeros(size(path, 1), 5); % HDR-VDP: Q, P - PUSSIM - pu_PSNR - log_PSNR
    IQA_raw = zeros(size(path, 1), 3); %  PU_MS_SSIM - pu_PSNR - log_PSNR
    
    for i = 1 : size(path, 1)
        fprintf('%d image in processing.\n',i);
        if is_non_calibrated
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
        else
            if i == 9
                hdr = hdrread(path{i});
            else
                hdr = exrread(path{i});
            end
            hdr(hdr < 0) = 0;
            
            
            if i == 1
                E = hdr(1 : 32 * 43, 1 : 32 * 116, :); % Wate
            end
            
            if i == 2
                E = hdr(1 : 32 * 21, : , :); % mpi_atrium_1
            end
            
            if i == 3
                E = hdr(1 : 32 * 31, 1 : 32 * 23, :); % AtriumNight
            end
            
            if i == 4
                E = hdr; % Route 66
            end
            
            if i == 5
                E = hdr(1 : 32 * 75, : ,:); % Wall Drug
                E = E * 2260;
            end
            
            if i == 6
                E = hdr * 4;
            end
            
            if i == 7
                E = hdr * 140;
            end
            
            if i == 8
                E = hdr * 4690;
            end
            
            if i == 9
                E = hdr(1 : 1056, :, :); % C images
            end
            
            N = E;
        end
        
        gt = N;
        [m, n, ~] = size(gt);
        
        % Restored Outputs
        load(fullfile(tp, image_names{i}));
        output = HDR;

        % Differences map
        rgb_diff = abs(VDP(gt) - VDP(output)) / 4095.0;
        
        % HDR-VDP
        vdp = hdrvdp(output, gt, 'rgb-bt.709', ppd, { 'surround_l', mean(mean(mean(gt)))});
        pvalue = sum(sum(vdp.P_map)) / (m * n);
        
        PMap = hdrvdp_visualize( 'pmap', vdp.P_map, { 'context_image', gt });
        
        % PU-MS-SSIM
        pu_ms_ssim = qm_pu2_msssim(output, gt);
 
        % PU psnr
        pu_psnr = qm_pu2_psnr(output, gt);

        % LOG psnr
        log_psnr = qm_log_psnr_rgb(output, gt);
        
        % Write Images
        imwrite(PMap, strcat(sp_prob, image_names{i}(1 : end - 4), '.png'));
        imwrite(GammaTMO(ReinhardTMO(output, 0.18), 2.2, 0, 0), strcat(sp_tone, image_names{i}(1 : end - 4), '.png'));
        imwrite(GammaTMO(ReinhardTMO(gt, 0.18), 2.2, 0, 0), strcat(sp_tone, image_names{i}(1 : end - 4), '_gt.png'));
        
        imwrite(rgb_diff * 200, strcat(sp_diff, image_names{i}(1 : end - 4), '_diff.png'));
        imwrite(rgb_diff(:,:,1) * 200, strcat(sp_diff, image_names{i}(1 : end - 4), '_diff_R.png'));
        imwrite(rgb_diff(:,:,2) * 200, strcat(sp_diff, image_names{i}(1 : end - 4), '_diff_G.png'));
        imwrite(rgb_diff(:,:,3) * 200, strcat(sp_diff, image_names{i}(1 : end - 4), '_diff_B.png'));
        
        % Write value: HDR-VDP: Q, P - HDR-VQM - PUSSIM - pu_PSNR - log_PSNR
        IQA(i, 1) = pvalue;
        IQA(i, 2) = vdp.Q;

        IQA(i, 3) = pu_ms_ssim;
        IQA(i, 4) = pu_psnr;
        IQA(i, 5) = log_psnr;
        
    end

    save(strcat(sp_IQA, 'IQA.mat'), 'IQA');   
    
    avg_Pvalue = mean(IQA(:, 1))
    avg_Qvalue = mean(IQA(:, 2))

    avg_MSSSIM = mean(IQA(:, 3))
    avg_PU = mean(IQA(:, 4))
    avg_LOG = mean(IQA(:, 5))
    

end
