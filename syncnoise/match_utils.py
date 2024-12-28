import os
from matplotlib import cm
import imageio
import cv2
import numpy as np
import torch


def meshgrid2d(b, y, x, device='cuda'):
    grid_y = torch.linspace(0.0, y-1, y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, y, 1])
    grid_y = grid_y.repeat(b, 1, x)
    grid_x = torch.linspace(0.0, x-1, x, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, x])
    grid_x = grid_x.repeat(b, y, 1)
    return grid_y, grid_x


@torch.no_grad()
def match_by_pips(
    pips_model,
    image_seq,
    height,
    width,
    grid_size=64,
    grid_batch_size=256,
    img_batch_size=8,
):
    """
    image_seq: value in [0,1]
    """
    b, t, c, ori_h, ori_w = image_seq.shape
    image_seq = image_seq.reshape([b*t, c, ori_h, ori_w])
    image_seq = torch.nn.functional.interpolate(image_seq, (height, width), mode='bilinear')
    image_seq = image_seq.reshape([b, t, c, height, width])

    grid_y, grid_x = meshgrid2d(1, grid_size, grid_size, device='cuda')
    grid_y = 8 + grid_y.reshape(1, -1)/float(grid_size-1) * (height-16)
    grid_x = 8 + grid_x.reshape(1, -1)/float(grid_size-1) * (width-16)
    xy = torch.stack([grid_x, grid_y], dim=-1) # 1, grid_size*grid_size, 2
    traj = torch.zeros([b, t, grid_size*grid_size, 2]).to(xy)
    with torch.no_grad():
        i = 0
        while i < xy.shape[1]:
            if i+grid_batch_size > xy.shape[1]:
                grid_batch_size = xy.shape[1] - i
            j = 0
            img_bs = img_batch_size
            while j < b:
                if j+img_bs > b:
                    img_bs = b - j
                preds = pips_model(xy[:,i:i+grid_batch_size].repeat(img_bs,1,1), 
                                   image_seq[j:j+img_bs]*255, iters=6)[0]
                traj[j:j+img_bs,:,i:i+grid_batch_size,:] = preds[-1]
                j += img_bs
            i += grid_batch_size

    # new_traj = torch.zeros_like(traj)
    # new_traj[...,0] = traj[...,0] / float(ori_w) * float(width)
    # new_traj[...,1] = traj[...,1] / float(ori_h) * float(height)
    new_traj = traj
    new_traj_flat = new_traj[...,1]*width + new_traj[...,0]

    return new_traj, new_traj_flat, image_seq


def gather_and_avg_by_traj(latents, traj_flat, invalid_mask=None, index=0, cluster_func=lambda x,y: x, use_cluster=True):
    """Gather latents at traj_flat and average them.
    Args:
        latents (torch.Tensor): (t, c, h, w)
        traj_flat (torch.Tensor): (t, h*w)
        cluster_func: a function takes in ori, avg latents, and returns a new latents
    Returns:
        latents (torch.Tensor): (t, c, h, w)
    """
    t, c, h, w = latents.shape
    latents_flat = latents.flatten(2,3).permute(1,0,2) # (c, t, h*w)
    traj_flat = traj_flat.unsqueeze(0).expand(c,-1,-1)
    invalid_mask = invalid_mask.unsqueeze(1).expand(-1,c,-1).permute(1,0,2).bool()
    latents_traj = torch.gather(latents_flat, 2, traj_flat) # (c, t, h*w)
    latents_traj_center = (latents_traj*(~invalid_mask)).sum(1, keepdim=True)/(~invalid_mask).sum(1, keepdim=True).expand_as(latents_traj) # (c, t, h*w)
    if use_cluster:
        latents_traj_move = cluster_func(latents_traj, latents_traj_center)[:,index] # (c, t, h*w)
    else:
        latents_traj_move = latents_traj_center[:,index]

    # fill back
    # if invalid_mask is not None:
    #     if len(invalid_mask.shape) == 2:
    #         invalid_mask = invalid_mask.unsqueeze(1).expand(-1,c,-1).permute(1,0,2)  # (c,t,h*w)
    #     latents_traj_move = torch.where(invalid_mask[:,index], latents_traj[:,index], latents_traj_move)
    
    latents_flat[:,index] = torch.scatter(latents_flat[:,index].float(), 1, traj_flat[:,index], latents_traj_move) # (t, c, h*w)
    return latents_flat.permute(1,0,2).view(t, c, h, w)


def pixel_replace_by_traj(latents, mask, traj_flat, invalid_mask, wait_list, edited_list, index=0, cluster_func=lambda x,y: x, use_cluster=True):
    """Gather latents at traj_flat and average them.
    Args:
        latents (torch.Tensor): (t, c, h, w)
        traj_flat (torch.Tensor): (t, h*w)
        cluster_func: a function takes in ori, avg latents, and returns a new latents
    Returns:
        latents (torch.Tensor): (t, c, h, w)
    """
    t, c, h, w = latents.shape
    latents_flat = latents.flatten(2,3).permute(1,0,2) # (c, t, h*w)
    traj_flat = traj_flat.unsqueeze(0).expand(c,-1,-1)
    latents_traj = torch.gather(latents_flat, 2, traj_flat) # (c, t, h*w)
    latents_traj_center = latents_traj[:,index]

    mask_flat = mask.flatten(2,3).permute(1,0,2)
    mask_traj = torch.gather(mask_flat, 2, traj_flat[0].unsqueeze(0))

    # sorted_list = torch.argsort(invalid_mask.sum(1))[:8]
    # index = sorted_list[1].item()
    # for i in range(1,len(sorted_list)):
    #     if sorted_list[i].item() in edited_list:
    #         continue
    #     else:
    #         index = sorted_list[i].item() 
    #         break       

    indices = np.where((invalid_mask.sum(1) < invalid_mask.shape[1] * 0.2).cpu().numpy())[0].tolist()
    for index in indices:
        if (index not in edited_list) and (index not in wait_list):
            unique_elements, unique_indices = np.unique(traj_flat[0,index].cpu().numpy(), return_index=True)
            unique_indices = torch.from_numpy(unique_indices).to(traj_flat.device)
            latents_traj_move = torch.where(invalid_mask[index].expand(c, -1).bool(), latents_traj[:,index], latents_traj_center)
            mask_traj_move = torch.where(invalid_mask[index].bool(), mask_traj[:,index], 0)
            latents_flat[:,index] = torch.scatter(latents_flat[:,index], 1, traj_flat[:,index][:,unique_indices], latents_traj_move[:,unique_indices]) # (t, c, h*w)
            mask_flat[:,index] = torch.scatter(mask_flat[:,index], 1, traj_flat[0,index].unsqueeze(0)[:,unique_indices], mask_traj_move[:,unique_indices])
            wait_list.append(index)
            edited_list.append(index)
    return latents_flat.permute(1,0,2).view(t, c, h, w), mask_flat.permute(1,0,2).view(t, 1, h, w), wait_list, edited_list


def init_latents(latents_shape, pixel_coord_flat_list, invalid_mask_list, device='cuda', match=True):
    """
    traj_latent_flat (torch.Tensor): (n_group, t, h*w)
    """
    latents = torch.randn([*latents_shape]).to(device)
    if not match:
        return latents
    for i in range(len(pixel_coord_flat_list)):
        pixel_coord_flat = torch.roll(pixel_coord_flat_list[i], shifts=i, dims=0)
        invalid_mask = torch.roll(invalid_mask_list[i], shifts=i, dims=0)
        latents = gather_and_avg_by_traj(latents, pixel_coord_flat, invalid_mask, i, lambda x,y: y)
    return latents


def draw_traj_on_image_py(rgb, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
    """Draw traj on rgb. Function from PIPS.
    """
    # all inputs are numpy tensors
    # rgb is 3 x H x W
    # traj is S x 2
    
    H, W, C = rgb.shape
    assert(C==3)

    rgb = rgb.astype(np.uint8).copy()

    S1, D = traj.shape
    assert(D==2)

    color_map = cm.get_cmap(cmap)
    S1, D = traj.shape

    for s in range(S1-1):
        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[s]-traj[0])**2))/maxdist).clip(0,1)
            color = np.array(color_map(val)[:3]) * 255 # rgb
        else:
            color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255 # rgb

        cv2.line(rgb,
                    (int(traj[s,0]), int(traj[s,1])),
                    (int(traj[s+1,0]), int(traj[s+1,1])),
                    color,
                    linewidth,
                    cv2.LINE_AA)
        if show_dots:
            cv2.circle(rgb, (traj[s,0], traj[s,1]), linewidth, color, -1)

    if maxdist is not None:
        val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        color = np.array(color_map(val)[:3]) * 255 # rgb
    else:
        # draw the endpoint of traj, using the next color (which may be the last color)
        color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb
        
    # color = np.array(color_map(1.0)[:3]) * 255
    cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)

    return rgb


def visualize_match(trajs, rgbs, show_dots=True, cmap='coolwarm', linewidth=1, save_path=None):
    """Visualize trajs on rgbs. Function from PIPS.

    Args:
        trajs (torch.Tensor): (B, S, N, 2)
        rgbs (torch.Tensor): (B, S, C, H, W)

    """
    # trajs is B, S, N, 2
    # rgbs is B, S, C, H, W
    B, S, C, H, W = rgbs.shape
    B, S2, N, D = trajs.shape
    assert(S==S2)

    rgbs = rgbs[0] # S, C, H, W
    trajs = trajs[0] # S, N, 2

    rgbs_color = []
    rgbs_draw = []
    for rgb in rgbs:
        rgb = (rgb/2+1).detach().cpu().numpy()*255 # 3 x H x W
        rgb = np.transpose(rgb, [1, 2, 0]).astype('uint8') # put channels last
        rgbs_color.append(rgb)
        rgbs_draw.append(rgb.copy())

    for i in range(N):
        traj = trajs[:,i].long().detach().cpu().numpy() # S, 2
        for t in range(S):
            rgbs_draw[t] = draw_traj_on_image_py(rgbs_draw[t], traj[:t+1], S=S,
                                                 show_dots=show_dots, cmap=cmap, linewidth=linewidth)

    if save_path is not None:
        os.makedirs(f'{save_path}/image_buffer', exist_ok=True)
        for t in range(S):
            imageio.imwrite(f'{save_path}/image_buffer/{t:02d}.jpg', rgbs_draw[t])
            imageio.imwrite(f'{save_path}/image_buffer/ori_{t:02d}.jpg', rgbs_color[t])
        imageio.mimwrite(f'{save_path}/traj.mp4', rgbs_draw, fps=8)
        imageio.mimwrite(f'{save_path}/ori_traj.mp4', rgbs_color, fps=8)

    return rgbs_draw, rgbs_color
