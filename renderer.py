import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    # TEST 
    mem_usage = 0
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        # TEST
        allocated = torch.cuda.memory_allocated(device)
        mem_usage += allocated

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    tensorf.recordStorage(mem_usage)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        # # TEST
        # print('* Rays: ', type(rays), rays.size())
        # print(rays, '\n')

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        # # TEST
        # print('* RGB map: ', type(rgb_map), rgb_map.size())
        # print(rgb_map, '\n')
        # print('* Depth map: ', type(depth_map), depth_map.size)
        # print(depth_map, '\n')


        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))    

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

# TEST 
# Obtain wanted uneven voxel grid by this criterion
@torch.no_grad()
def uneven_critrn(test_dataset, tensorf, res_target, args, renderer, N_vis=5, N_samples=-1,
                    white_bg=False, ndc_ray=False, device='cuda'):
    init_N_voxels = tensorf.getTotalVoxels()
    # each voxel will include 1 opacity + 3 appearances for color
    init_mem = init_N_voxels * 4 * 24  # bytes
    n_select_to_comp = -1
    select_count = 0
    n_read = 0    
    VF_lambda = args.VF_lambda  # 實驗定值

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)

    n_imgs_ref = test_dataset.get_test_n_img(getitem=True)
    n_rays_ref = len(test_dataset.all_rgbs)
    # n_select_to_comp = int(0.1 * n_imgs_ref)   # 10% of dataset images & 10 at least   
    # assert n_select_to_comp >= 5 

    # if n_select_to_comp < 10:
    #     n_select_to_comp = 10
    n_select_to_comp = 10    # 實驗：scaled_down固定取10張（/wineholder scene）

    print(f'... to select {n_select_to_comp} images at different viewpoints.')

    pixel_per_img = n_rays_ref / n_imgs_ref   # 每張圖擁有的pixel數目，也等於採樣ray數目（全採樣）

    N_partit, N_check, check_coords = tensorf.uneven_info()
    mask_pend = None
    prev_mem = init_mem

    for n in range(N_check):
        try_target = check_coords[n,...]
        try_pass = False
        
        idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
        for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
            W, H = test_dataset.img_wh
            rays = samples.view(-1,samples.shape[-1])

            if (idx + 1) % pixel_per_img == 0:
                n_read += 1

            if n_read % 10 != 0:    # 實驗：scaled_down固定取10張（/wineholder scene）
                continue
            if select_count > n_select_to_comp:  # selection is done
                break
            
            select_count += 1
            """
            Criterion for uneven voxel partition:
                R + lambda * D   （concept from Rate-distortion optmization）   
                =>  potential = MSE + VF_lambda * grid_storage, where 0 < VF_lambda <= 1.
            """            
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            
            # focus the target region and record the uneven grid 
            if mask_pend is None:
                partit_mask, prev_mask, try_mask =  tensorf.subgrid_shift(N_partit, try_target, mask_pend=mask_pend, keep_also=False, new_case=True)
                coar_density_plane, coar_density_line, coar_app_plane, coar_app_line = tensorf.getVM()
                mapped_density_plane, mapped_density_line, intrp_density_plane, intrp_density_line = tensorf.upsample_VM(coar_density_plane, coar_density_line, res_target)
                mapped_app_plane, mapped_app_line, intrp_app_plane, intrp_app_line = tensorf.upsample_VM(coar_app_plane, coar_app_line, res_target)

            elif try_pass:
                partit_mask, prev_mask, try_mask =  tensorf.subgrid_shift(N_partit, try_target, mask_pend=mask_pend, keep_also=True, new_case=False)
            else:
                partit_mask, prev_mask, try_mask =  tensorf.subgrid_shift(N_partit, try_target, mask_pend=mask_pend, keep_also=False, new_case=False)

            # prediction by prev grid
            # tensorf.resetMemCount()
            if mask_pend is None:
                prev_density_plane, prev_density_line = mapped_density_plane, mapped_density_line
                prev_app_plane, prev_app_line = mapped_app_plane, mapped_app_line
            else:
                prev_density_plane, prev_density_line, prev_app_plane, prev_app_line = tensorf.getVM()
            tensorf.setVM(prev_density_plane, prev_density_line, prev_app_plane, prev_app_line)

            prev_rgb_map, _, prev_depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                                            ndc_ray=ndc_ray, white_bg = white_bg, device=device)
            prev_rgb_map = prev_rgb_map.clamp(0.0, 1.0)
            prev_rgb_map, prev_depth_map = prev_rgb_map.reshape(H, W, 3).cpu(), prev_depth_map.reshape(H, W).cpu()
            
            prev_MSE = torch.mean((prev_rgb_map - gt_rgb) ** 2)
            # prev_mem = tensorf.getTotalStorage()


            # prediction by new grid
            # tensorf.resetMemCount()
            new_density_plane, new_density_line = tensorf.form_uneven_grid_VM(prev_density_plane, prev_density_line, intrp_density_plane, intrp_density_line, res_target, partit_mask)
            new_app_plane, new_app_line = tensorf.form_uneven_grid_VM(prev_app_plane, prev_app_line,intrp_app_plane,intrp_app_line, res_target, partit_mask)
            tensorf.setVM(new_density_plane, new_density_line, new_app_plane, new_app_line)

            try_rgb_map, _, try_depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                                            ndc_ray=ndc_ray, white_bg = white_bg, device=device)
            try_rgb_map = try_rgb_map.clamp(0.0, 1.0)
            try_rgb_map, try_depth_map = try_rgb_map.reshape(H, W, 3).cpu(), try_depth_map.reshape(H, W).cpu()
            
            try_MSE = torch.mean((try_rgb_map - gt_rgb) ** 2)
            try_mem = prev_mem + 24 * 4   # bytes
            # try_mem = tensorf.getTotalStorage()

            # print(f' * prev_mem:{prev_mem} , try_mem:{try_mem}\n')

            # calculate potential value    
            prev_poten = prev_MSE + VF_lambda * prev_mem
            try_poten = try_MSE + VF_lambda * try_mem            


            if try_poten < prev_poten:
                try_pass = True
                mask_pend = try_mask
            else:
                try_pass = False
                mask_pend = prev_mask
                tensorf.setVM(prev_density_plane, prev_density_line, prev_app_plane, prev_app_line)
                break

        tensorf.setMask(mask_pend)  # 該區確定需要細切，記錄傳遞
        tensorf.addVoxel()
        prev_mem += 24 * 4  # bytes

        if n == (N_check-1):
            print('========> uneven grid is done.')
        elif try_pass:
            tensorf.update_unevenMask(mask_pend)
            tensorf.addVoxel()
        else:
            tensorf.update_unevenMask(mask_pend)


@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

