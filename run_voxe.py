import os, sys, copy, time, random, argparse
from tqdm import tqdm, trange
from skimage.transform import resize

import mmengine
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from lib import utils, dvgo, dcvgo, dmpigo, sd
from lib.load_data import load_data
import matplotlib.pyplot as plt

from torch_efficient_distloss import flatten_eff_distloss

def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def resize_image_batch(images, scale_factor):
    # Get the original image dimensions
    images = images.numpy()
    batch_size, original_height, original_width, channels = images.shape
    
    # Calculate the new dimensions
    new_height = int(original_height / scale_factor)
    new_width = int(original_width / scale_factor)
    
    # Initialize the resized image batch
    resized_images = np.empty((batch_size, new_height, new_width, channels), dtype=images.dtype)
    
    # Resize each image in the batch
    for i in range(batch_size):
        resized_images[i] = resize(images[i], (new_height, new_width), preserve_range=True, mode='constant')
    
    resized_images = torch.from_numpy(resized_images)
    return resized_images

def _density_correlation_loss(sds_density, regular_density):
    eps = 0.0000001 # for numerical stability

    # Calculate Denominator:
    sds_var = torch.mean((sds_density - torch.mean(sds_density))**2)
    regular_var = torch.mean((regular_density - torch.mean(regular_density))**2)
    denominator = torch.sqrt(sds_var * regular_var)

    # Calculate Covariance:
    covariance_grid = (sds_density - torch.mean(sds_density)) * \
        (regular_density - torch.mean(regular_density))
    #covariance = torch.mean(covariance_grid)

    # Return Result:
    correlation_grid = covariance_grid / (denominator + eps)
    correlation = torch.mean(correlation_grid)
    return 1.0 - correlation, correlation_grid.detach()

#####################################
####       VOX-E TRAINING        ####
#####################################


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_ckpt_path_voxe = os.path.join(cfg.basedir, cfg.expname, f'voxe_last.tar')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW_orig, Ks_orig, near, far, i_train, i_val, i_test, poses, render_poses, images_orig = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # init sds loss class
    sds_loss = sd.scoreDistillationLoss(device, 
                                        args.voxe_prompt, 
                                        t_sched_start = cfg_train.voxe_sds_t_start,
                                        t_sched_freq = cfg_train.voxe_sds_t_freq,
                                        t_sched_gamma = cfg_train.voxe_sds_t_gamma,
                                        directional = False)

    images = resize_image_batch(images_orig, cfg_model.scale_factor_voxe)
    HW = (HW_orig/cfg_model.scale_factor_voxe).astype(np.uint32)
    Ks = Ks_orig / cfg_model.scale_factor_voxe

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last.tar')
    print(f'scene_rep_reconstruction (fine): reload from {last_ckpt_path}')
    model, optimizer, start = load_existed_model(args, cfg, cfg_train, last_ckpt_path)
    start = 0

    # make copy of model here
    ref_model = copy.deepcopy(model)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize_voxe,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    # ES Addition - lower the LR a bit:
    for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * 0.8

    for global_step in trange(1+start, 1+cfg_train.N_iters_voxe):

        # progress scaling checkpoint
        #if global_step in cfg_train.pg_scale:
        #    n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
        #    cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
        #    if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
        #        model.scale_volume_grid(cur_voxels)
        #    elif isinstance(model, dmpigo.DirectMPIGO):
        #        model.scale_volume_grid(cur_voxels, model.mpi_depth)
        #    else:
        #        raise NotImplementedError
        #    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        #    model.act_shift -= cfg_train.decay_after_scale
        #    torch.cuda.empty_cache()
        
        sel_i = torch.randint(rgb_tr.shape[0], (1,))
        target = rgb_tr[sel_i].reshape((-1, 3)) # TODO (ES): remove reshape later
        rays_o = rays_o_tr[sel_i].reshape((-1, 3))
        rays_d = rays_d_tr[sel_i].reshape((-1, 3))
        viewdirs = viewdirs_tr[sel_i].reshape((-1, 3))

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)
        
        # output render results (test)
        if global_step % cfg_train.freq_test_voxe == 0:
            rendered_img = render_result['rgb_marched'].reshape((HW[0, 0], HW[0, 1], 3))
            rendered_img = rendered_img.detach().cpu().numpy()
            im_path_output = os.path.join(cfg.basedir, cfg.expname, f'test_output_{global_step}.png')
            plt.imsave(im_path_output, rendered_img)

            rendered_img_target = target.reshape((HW[0, 0], HW[0, 1], 3))
            rendered_img_target = rendered_img_target.detach().cpu().numpy()
            im_path_target = os.path.join(cfg.basedir, cfg.expname, f'test_target_{global_step}.png')
            plt.imsave(im_path_target, rendered_img_target)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        
        # SDS Loss:
        loss = sds_loss.training_step(render_result['rgb_marched'],
                                      HW[0, 0], HW[0, 1],
                                      global_step=global_step)

        # DCL:
        dcl_loss, _ = _density_correlation_loss(model.density.grid, ref_model.density.grid.detach())
        loss = loss + cfg_train.voxe_DCL_weight * dcl_loss

        #loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr 
        # TODO (ES): Figure out if we want to use this
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step % args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'DCL: {dcl_loss.item():.9f} /'
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step % args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path_voxe)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path_voxe)


def train_voxe(args, cfg, data_dict):
    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0: # TODO (ES): change from iters to load / don't load coarse geo
        #scene_rep_reconstruction(
        #        args=args, cfg=cfg,
        #        cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
        #        xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
        #        data_dict=data_dict, stage='coarse')
        #eps_coarse = time.time() - eps_coarse
        #eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        #print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0: # TODO (ES): change from iters to load / don't load coarse geo
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')