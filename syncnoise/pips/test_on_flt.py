import time
import numpy as np
import timeit
import saverloader
from nets.raftnet import Raftnet
from nets.pips import Pips
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from flyingthingsdataset import FlyingThingsDataset
import utils.basic
import utils.improc
import utils.test
from fire import Fire

device = 'cuda'
random.seed(125)
np.random.seed(125)

def run_dino(dino, d, sw):
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    B, S1, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)
    
    # compute per-sequence visibility labels
    vis_g = (torch.sum(vis_g, dim=1, keepdim=True) >= 4).float().repeat(1, S, 1)

    _, S, C, H, W = rgbs.shape

    trajs_e = utils.test.get_dino_output(dino, rgbs, trajs_g, vis_g)

    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))

        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)
    
    return metrics


def run_pips(model, d, sw):

    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)

    # compute per-sequence visibility labels
    vis_g = (torch.sum(vis_g, dim=1, keepdim=True) >= 4).float().repeat(1, S, 1)

    _, S, C, H, W = rgbs.shape

    preds, preds_anim, vis_e, stats = model(trajs_g[:,0], rgbs, iters=6, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)

    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    trajs_e = preds[-1]

    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)
        
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)

        # animate_traj2ds_on_rgbs
        rgb_vis = []
        black_vis = []
        for trajs_e_ in preds_anim:
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], gt_rgb, only_return=True, cmap='coolwarm'))
            black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], gt_black, only_return=True, cmap='coolwarm'))
        sw.summ_rgbs('outputs/animated_trajs_on_black', black_vis)
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)
        
    return metrics


def run_raft(raft, d, sw):
    rgbs = d['rgbs'].cuda().float() # B, S, C, H, W
    occs = d['occs'].cuda().float() # B, S, 1, H, W
    masks = d['masks'].cuda().float() # B, S, 1, H, W
    trajs_g = d['trajs'].cuda().float() # B, S, N, 2
    vis_g = d['visibles'].cuda().float() # B, S, N
    valids = d['valids'].cuda().float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)
    
    # compute per-sequence visibility labels
    vis_g = (torch.sum(vis_g, dim=1, keepdim=True) >= 4).float().repeat(1, S, 1)

    _, S, C, H, W = rgbs.shape

    prep_rgbs = utils.improc.preprocess_color(rgbs)

    flows_e = []
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        rgb1 = prep_rgbs[:,s+1]
        flow, _ = raft(rgb0, rgb1, iters=32)
        flows_e.append(flow)
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    coord0 = trajs_g[:,0] # B, N, 2
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2
    
    ate = torch.norm(trajs_e - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }
    
    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))

        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)

    return metrics


def main(
        exp_name='flt', 
        B=1,
        S=8,
        N=16,
        modeltype='pips',
        init_dir='reference_model',
        stride=4,
        log_dir='logs_test_on_flt',
        dataset_location='/data/flyingthings',
        max_iters=0, # auto-select based on dataset
        log_freq=100,
        shuffle=False,
        subset='all',
        crop_size=(384,512), # the raw data is 540,960
        use_augs=False,
):
    # the idea in this file is to evaluate on flyingthings++

    assert(modeltype=='pips' or modeltype=='raft' or modeltype=='dino')
    
    ## autogen a name
    model_name = "%d_%d_%d_%s" % (B, S, N, modeltype)
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    test_dataset = FlyingThingsDataset(
        dataset_location=dataset_location,
        dset='TEST', subset=subset,
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=24,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    test_iterloader = iter(test_dataloader)

    global_step = 0

    if modeltype=='pips':
        model = Pips(S=S, stride=stride).cuda()
        _ = saverloader.load(init_dir, model)
        model.eval()
    elif modeltype=='raft':
        model = Raftnet(ckpt_name='../../RAFT/models/raft-things.pth').cuda()
        model.eval()
    elif modeltype=='dino':
        patch_size = 8
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size).cuda()
        model.eval()
    else:
        assert(False) # need to choose a valid modeltype
        
    n_pool = 10000
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = utils.misc.SimplePool(n_pool, version='np')

    if max_iters==0:
        max_iters = len(test_dataloader)
    print('setting max_iters', max_iters)
    
    while global_step < max_iters:
        
        read_start_time = time.time()

        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        gotit = (False,False)
        while not all(gotit):
            try:
                sample, gotit = next(test_iterloader)
            except StopIteration:
                test_iterloader = iter(test_dataloader)
                sample, gotit = next(test_iterloader)

        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        with torch.no_grad():
            if modeltype=='pips':
                metrics = run_pips(model, sample, sw_t)
            elif modeltype=='raft':
                metrics = run_raft(model, sample, sw_t)
            elif modeltype=='dino':
                metrics = run_dino(model, sample, sw_t)
            else:
                assert(False) # need to choose a valid modeltype

        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])
        sw_t.summ_scalar('pooled/ate_all', ate_all_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_vis', ate_vis_pool_t.mean())
        sw_t.summ_scalar('pooled/ate_occ', ate_occ_pool_t.mean())

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f, ate_vis = %.2f, ate_occ = %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            ate_vis_pool_t.mean(), ate_occ_pool_t.mean()))

    writer_t.close()

if __name__ == '__main__':
    Fire(main)
