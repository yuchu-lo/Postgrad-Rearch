import configargparse
import math

def _int_nounderscore(x):
    return int(str(x).replace('_',''))

def _parse_bool(x):
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ('1','true','t','yes','y','on'):  return True
    if s in ('0','false','f','no','n','off'): return False
    raise configargparse.ArgumentTypeError(f'invalid boolean value: {x}')

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()    
    # ===== General / Run =====
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--expname', type=str,
                        help='experiment name')
    parser.add_argument('--basedir', type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument('--add_timestamp', type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument('--progress_refresh_rate', type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument('--enable_mem_breakdown_jsonl', action='store_true',
                        help='append detailed memory breakdown to mem_breakdown.jsonl at heartbeat (off by default)')

    # ===== Data =====
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['blender', 'llff', 'nsvf', 'tankstemple', 'own_data'],
                        help='dataset name')
    parser.add_argument('--datadir', type=str, default='./data/Synthetic_NSVF/Wineholder',
                        help='input data directory')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--with_depth', action='store_true',
                        help='assume ground truth depth is available for Nerf_plusplus')
    parser.add_argument('--ndc_ray', action='store_true',
                        help='use normalized device coordinates rays')
    parser.add_argument('--train_view', type=int, default=-1,
                        help='1~N: only use this view as training')
    parser.add_argument('--test_view', type=int, default=-1,
                        help='1~N: only use this view as testing')
    parser.add_argument('--downsample_train', type=float, default=1.0,
                        help='downsample ratio for training images')
    parser.add_argument('--downsample_test', type=float, default=1.0,
                        help='downsample ratio for test images')

    # ===== Model / Rendering =====
    parser.add_argument('--model_name', type=str, default='TensorVMSplitPatch')
    parser.add_argument('--shadingMode', type=str, default='MLP_Fea',
                        choices=['MLP_PE', 'MLP_Fea', 'MLP_FeaNG', 'MLP', 'SH', 'RGB'], 
                        help='which shading mode to use')
    parser.add_argument('--pos_pe', type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument('--view_pe', type=int, default=6,
                        help='number of pe for view')
    parser.add_argument('--fea_pe', type=int, default=6,
                        help='number of pe for features')
    parser.add_argument('--featureC', type=int, default=128,
                        help='hidden feature channel in MLP')
    parser.add_argument('--fea2denseAct', type=str, default='softplus',
                        choices=['softplus', 'relu', 'sigmoid'])
    parser.add_argument('--step_ratio', type=float, default=0.5,
                        help='sampling step size proportional to voxel size')
    parser.add_argument('--distance_scale', type=float, default=25,
                        help='scaling for distance transform / alpha')
    parser.add_argument('--density_shift', type=float, default=-10,
                        help='shift for density before activation')
    parser.add_argument('--rm_weight_mask_thre', type=float, default=0.0,
                        help='remove ray mask threshold (<=0 to disable)')
    
    # —— Shared basis —— 
    parser.add_argument('--global_basis_enable', type=int, default=1,
                    help='Enable global shared basis for VM decomposition (1=enabled, 0=use old method)')
    parser.add_argument('--global_basis_k_sigma', type=int, default=64,
                        help='Number of global basis vectors for density (higher=more expressive but more memory)')
    parser.add_argument('--global_basis_k_app', type=int, default=96,
                        help='Number of global basis vectors for appearance')

    # ===== Patch Continuity / Residual & Seam (Model toggles) =====
    parser.add_argument('--enable_child_residual', type=_parse_bool, default=True,
                        help='enable interior-gated residual per patch; residual=0 at faces to keep continuity')
    parser.add_argument('--residual_gate_tau', type=float, default=0.10,
                        help='interior width for residual gating in [0,1] (smaller = thinner interior; residual vanishes closer to faces)')
    parser.add_argument('--enable_seam_blend', type=_parse_bool, default=True,
                        help='enable lightweight seam blending near faces (no extra params or loss; smooths cross-patch boundaries)')
    parser.add_argument('--seam_band_width', type=float, default=0.05,
                        help='band width in local [0,1] from each face used for seam blending')

    # ===== Budgets / Reso Cap =====
    parser.add_argument('--voxel_budget', type=_int_nounderscore, default=30_000_000, 
                        help='global voxel budget hard cap')
    parser.add_argument('--vram_budget_MB', type=float, default=9000.0,                 
                        help='soft VRAM budget in MB')
    parser.add_argument('--N_voxel_init', type=_int_nounderscore, default=2_097_152,  
                        help='bootstrap voxel count (for schedulers)')
    parser.add_argument('--N_voxel_final', type=_int_nounderscore, default=27_000_000, 
                        help='target/final voxel count (for schedulers)')
    parser.add_argument('--vm_reso_max', type=int, default=64, 
                        help='max per-patch VM resolution (short side)')
    parser.add_argument('--patch_cap', type=int, default=96, 
                        help='max number of patches (safety cap)')

    # ===== Patches / VM =====
    parser.add_argument('--init_grid_res', nargs='*', type=int, default=[2, 2, 2],
                        help='initial patch grid res (Gx,Gy,Gz)')
    parser.add_argument('--init_vm_res', nargs='*', type=int, default=[8, 8, 8],
                        help='initial per-patch VM res (Rx,Ry,Rz)')
    parser.add_argument('--n_factors_sigma', nargs='*', type=int, default=None,
                        help='tensor decomposition rank for density planes/lines')
    parser.add_argument('--n_factors_app', nargs='*', type=int, default=None,
                        help='tensor decomposition rank for app planes/lines')
    parser.add_argument('--data_dim_color', type=int, default=27,
                        help='appearance feature dimension')
    parser.add_argument('--n_rgb', type=int, default=3,
                        help='rgb output dimension')
    
    # —— Rank resize —— 
    parser.add_argument('--dynamic_rank', type=_parse_bool, default=True,
                        help='enable per-patch dynamic rank resize (0/1/true/false)')
    parser.add_argument('--min_rank', type=int, default=8,
                        help='min rank for factorization (if used)')
    parser.add_argument('--max_rank', type=int, default=96,
                        help='max rank for factorization (if used)')
    parser.add_argument('--rank_min_res', type=int, default=8,
                        help='min reso threshold to allow to upgrade per-patch rank')
    parser.add_argument('--rank_verbose', type=int, default=1,
                        help='verbose prints for rank resize routines (0=quiet)')
    parser.add_argument('--rank_warmup', type=int, default=1500,
                        help='not attempt rank-upgrade before this iter (effective warmup can be trimmed before first event)')
    parser.add_argument('--rank_cooldown', type=int, default=800,
                        help='min gap between two rank upgrades')
    parser.add_argument('--rank_warmup_margin', type=int, default=200,
                        help='safety margin to allow first upgrade shortly before the first structural event')
    parser.add_argument('--rank_freeze_after', type=int, default=None,
                        help='stop rank upgrades after this iter if set')
    parser.add_argument('--rank_importance', type=str, default='alpha',
                        help='importance metric for rank upgrade/autoscale')
    parser.add_argument('--rank_down_after_iter', type=int, default=12000)
    parser.add_argument('--autoscale_cooldown_after_event', type=int, default=400)
    parser.add_argument('--selective_rank_autoscale', type=int, default=0,
                        help='enable autoscale + downsize ranks; value is ref_res (e.g., 128). 0 disables')
    parser.add_argument('--rank_autoscale_gamma', type=float, default=0.6,
                        help='autoscale exponent wrt min(res)/ref_res')
    parser.add_argument('--rank_autoscale_alpha_keep_q', type=float, default=0.85,
                        help='quantile of alpha_mass to protect from downsize')
    
    parser.add_argument('--rank_base_floor_sig', type=int, default=None)
    parser.add_argument('--rank_base_floor_app', type=int, default=None)
    parser.add_argument('--rank_floor_mode', type=str, default='steps', choices=['steps','scale'])
    parser.add_argument('--rank_floor_steps', type=str, default='')   # e.g., '16:16,48|32:24,72|64:32,96'
    parser.add_argument('--rank_floor_round_to', type=int, default=4)
    parser.add_argument('--rank_floor_scale_anchor', type=int, default=16)
    parser.add_argument('--rank_floor_scale_beta', type=float, default=0.5)

    parser.add_argument('--rank_cap_sigma', type=int, nargs='*', default=None,
                        help='hard cap for density rank per axis (int or 3 ints)')
    parser.add_argument('--rank_cap_app', type=int, nargs='*', default=None,
                        help='hard cap for app rank per axis (int or 3 ints)')
    parser.add_argument('--rank_budget_mb', type=float, default=0.0,
                        help='total memory budget (MB) for upgrading rank (no limits if <=0)')

    parser.add_argument('--repair_enable', action='store_true',
                        help='enable patch repair steps')
    parser.add_argument('--repair_mode', type=str, default='conservative',
                        choices=['off', 'conservative', 'full'],
                        help='repair policy')
    parser.add_argument('--repair_warmup', type=int, default=500)
    parser.add_argument('--repair_tau', type=float, default=0.0,
                        help='repair threshold (<=0 derive from stats)')
    parser.add_argument('--repair_adjacent_only', action='store_true',
                        help='only snap to adjacent patches')
    parser.add_argument('--repair_grad_scale_sigma', type=float, default=1.0,
                        help='gradient scale for density repair')
    parser.add_argument('--repair_grad_scale_app', type=float, default=1.0,
                        help='gradient scale for app repair')

    # ===== Training / Optim =====
    parser.add_argument('--n_iters', type=int, default=30000,
                        help='total training iterations')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch size (rays)')
    parser.add_argument('--lr_init', type=float, default=0.02,
                        help='initial learning rate for planes/lines')
    parser.add_argument('--lr_basis', type=float, default=0.001,
                        help='initial learning rate for basis')
    parser.add_argument('--lr_decay_iters', type=int, default=20000,
                        help='iterations for lr decay (<=0 -> use n_iters)')
    parser.add_argument('--lr_decay_target_ratio', type=float, default=0.1,
                        help='target lr ratio at decay end')

    # ===== Regularization (L1/TV/Ortho) =====
    parser.add_argument('--Ortho_weight', type=float, default=0.0,
                        help='orthogonal regularization weight')
    parser.add_argument('--L1_weight_inital', type=float, default=0.0,
                        help='initial L1 weight (32^3 stage)')
    parser.add_argument('--L1_weight_rest', type=float, default=0.0,
                        help='L1 weight after decay at 64^3 stage')
    parser.add_argument('--TV_weight_density', type=float, default=0.0,
                        help='TV weight on density')
    parser.add_argument('--TV_weight_app', type=float, default=0.0,
                        help='TV weight on appearance')
    parser.add_argument('--TV_app_bump_at64', type=float, default=1.25,
                        help='appearance TV bump factor at 64^3 stage')
    parser.add_argument('--REG_BURNIN_32', type=int, default=2000,
                        help='burn-in (iter) before 32^3 reg warmup')
    parser.add_argument('--REG_BURNIN_64', type=int, default=6000,
                        help='burn-in (iter) before 64^3 reg stage')
    parser.add_argument('--reg_warmup_iters', type=int, default=1500,
                        help='reg warmup iterations')
    parser.add_argument('--reg_decay_iters', type=int, default=1500,
                        help='reg decay iterations')
    parser.add_argument('--tv_res_gate', type=int, default=32,
                        help='TV enable gate by min res')
    parser.add_argument('--tv_force_bump_after', type=int, default=-1,
                        help='force TV bump after this iter (-1=disable)')

    # ===== Alpha mask / Pruning =====
    parser.add_argument('--alpha_mask_thre', type=float, default=0.001)
    parser.add_argument('--alpha_mask_base_res', type=int, default=192,
                        help='base alpha grid res (on chosen aspect)')
    parser.add_argument('--alpha_mask_min_res', type=int, default=64,
                        help='min alpha grid res per axis')
    parser.add_argument('--alpha_mask_max_res', type=int, default=512,
                        help='max alpha grid res per axis')
    parser.add_argument('--alpha_mask_aspect', type=str, default='short',
                        choices=['short', 'long', 'mean'],
                        help='which aabb edge defines the base scaling')
    parser.add_argument('--alpha_freeze_after_event', type=int, default=400,
                        help='iterations to freeze alpha update after structure events')

    # ===== Coverage / Heartbeat =====
    parser.add_argument('--heartbeat_every', type=int, default=250, 
                        help='run heartbeat every N iters')
    parser.add_argument('--heartbeat_target_miss', type=float, default=0.15,
                        help='target missing ratio for coverage guard')
    parser.add_argument('--heartbeat_seed_cells', type=int, default=8,
                        help='seed cells to add when guarding coverage')
    parser.add_argument('--miss_ratio_samples', type=int, default=8192,
                        help='samples for quick miss ratio estimation')
    parser.add_argument('--miss_guard_thres', type=float, default=0.35,
                        help='threshold to trigger aggressive fill')

    # ===== Uneven criterion =====
    parser.add_argument('--critrn_mode', type=str, default='hybrid',
                        choices=['split', 'vm', 'hybrid'],
                        help='refinement mode: split-only, VM-upgrade-only, or hybrid')
    parser.add_argument('--critrn_every', type=int, default=1500, 
                        help='apply uneven criterion every N iterations (reduce for faster training)')
    parser.add_argument('--critrn_vm_topk', type=int, default=8,
                        help='max number of patches to VM-upgrade per criterion pass')
    parser.add_argument('--critrn_split_topk', type=int, default=4,
                        help='legacy cap for split candidates (used as safety; hybrid will combine with refine_frac)')
    parser.add_argument('--critrn_focus_start', type=int, default=8000,
                        help='start iteration to use focus sampling around activated patches')
    parser.add_argument('--critrn_focus_halo', type=int, default=1,
                        help='Chebyshev halo in patch-grid for focus sampling (neighbors in +/- halo)')
    parser.add_argument('--critrn_focus_samples', type=int, default=4,
                        help='per-ray samples used to build focus map')
    parser.add_argument('--critrn_min_total_rays', type=int, default=2000,
                        help='min number of rays aggregated across views for a patch to be evaluated')
    parser.add_argument('--critrn_global_mix_ratio', type=float, default=0.3,
                        help='fraction of non-focus rays mixed with focus rays per view')
    parser.add_argument('--critrn_sample_rays', type=int, default=1024,
                        help='number of rays to sample per view for criterion evaluation')
    parser.add_argument('--critrn_vm_metric', type=str, default='gain_per_mem',
                        choices=['gain_per_mem', 'margin'],
                        help='ranking metric for VM upgrades')
    parser.add_argument('--critrn_vm_floor_min', type=int, default=0, metavar='RES',
                        help='VM res floor; prioritize patches with min(res) < RES (0=off)')
    parser.add_argument('--critrn_vm_floor_share', type=float, default=0.35, metavar='RATIO',
                        help='fraction of VM slots reserved for floor upgrades')
    parser.add_argument('--critrn_vm_floor_min_count', type=int, default=1, metavar='N',
                        help='minimum VM slots reserved for floor upgrades')

    # —— VM structual cost —— 
    parser.add_argument('--critrn_lambda', type=float, default=0.5,
                        help='weighting factor in Patch Utility Function (PUF): P = MSE + λ*norm_memory')
    parser.add_argument('--logmargin_base', type=float, default=None,
                        help='margin ratio used to compute log-margin threshold τ (overrides --logmargin_tau if provided)')
    parser.add_argument('--logmargin_tau', type=float, default=math.log(1.05),
                        help='log margin threshold τ (used only if --logmargin_base is not given)')
    parser.add_argument('--critrn_accept_ratio', type=float, default=0.5,
                        help='min. fraction of views that must agree to mark a patch as candidate')
    parser.add_argument('--critrn_refine_frac', type=float, default=0.3,
                        help='fraction of candidate patches to actually split')
    
    # —— Boundary regularization cost —— 
    parser.add_argument('--puf_alpha_boundary', type=float, default=0.3,
                        help='weighting factor of boundary capacity cost in PUF for splits; 0.0 disables')
    parser.add_argument('--boundary_cost_mode', type=str, default='dof', 
                        choices=['dof', 'const'],
                        help='boundary cost proxy: dof ~ ry*rz + rx*rz + rx*ry; const = 1')
    parser.add_argument('--boundary_smooth_strength', type=float, default=1.0,
                        help='scale applied to (1 - roughness_proxy); higher => stronger penalty in smooth areas')

    # ===== Split / VM-upsample / Re-filter =====
    parser.add_argument('--rebucket_at', type=int, default=-1,
                        help='refilter (annealed mixing) trigger iteration; -1 disables (default)')
    parser.add_argument('--refilter_at', type=int, dest='rebucket_at',
                        help='alias of --rebucket_at (annealed refilter trigger)')
    parser.add_argument('--refilter_samples', type=int, default=256,
                        help='samples per ray when computing validity during re-filter')
    parser.add_argument('--refilter_anneal_iters', type=int, default=1500,
                        help='anneal window (iters) for mixed sampler new_ratio ramp')
    parser.add_argument('--refilter_start_ratio', type=float, default=0.80,
                        help='initial fraction of new-pool samples at refilter start (ramps to 1.0)')

    # —— Split-time local upgrades ——
    parser.add_argument('--child_micro_ups_enable', type=_parse_bool, default=True,
                        help='enable per-child micro upsample right after an accepted split (local VM res only, capped by vm_reso_max)')
    parser.add_argument('--child_micro_ups_scale', type=int, default=2,
                        help='per-axis scale factor for micro upsample on new children (e.g., 2 -> Rx*2,Ry*2,Rz*2, capped)')
    parser.add_argument('--child_lr_boost_enable', type=_parse_bool, default=True,
                        help='temporarily boost learning rate for newly created child parameters after split')
    parser.add_argument('--child_lr_boost_mult', type=float, default=1.5,
                        help='multiplier for LR of new child params during the boost horizon (e.g., 1.5)')
    parser.add_argument('--child_lr_boost_iters', type=int, default=300,
                        help='iterations to keep the LR boost active for new child params (will decay back to base)')

    # —— Strict-even split —— 
    parser.add_argument('--strict_even_split', action='store_true',
                        help='enable strict even split (G -> n^3)')
    parser.add_argument('--strict_even_n', type=int, default=3,
                        help='target n for strict-even split')
    parser.add_argument('--skip_strict_even', action='store_true',
                        help='skip strict-even split')
    
    # —— Splitting to be uneven —— 
    parser.add_argument('--split_boost_enable', type=_parse_bool, default=False,
                        help='enable optional child-boost after split (0/1)')
    parser.add_argument('--split_boost_start', type=int, default=12000,
                        help='start iter for split-boost')
    parser.add_argument('--split_boost_topq', type=float, default=0.20,
                        help='quantile of children (by alpha_mass) to boost')
    parser.add_argument('--split_boost_factor', type=float, default=1.5,
                        help='per-axis VM res scale factor for boosted children (capped by vm_reso_max)')
    parser.add_argument('--split_boost_vm_cap', type=int, default=8,
                        help='max number of children to boost per round')
    parser.add_argument('--split_at', type=int, default=0,
                        help='iteration to start coarse split (0=disable)')
    parser.add_argument('--coarse_split', action='store_true',
                        help='enable coarse split stage')
    parser.add_argument('--coarse_split_every', type=int, default=500,
                        help='coarse split interval')
    parser.add_argument('--split_child_res_policy', type=str, default='half',
                        choices=['arg', 'half', 'scale'],
                        help='child patch VM res policy')
    parser.add_argument('--split_child_min', type=int, default=16,
                        help='min child VM res')
    parser.add_argument('--upsample_when', type=str, default='never',
                        choices=['never', 'after_split', 'fixed'],
                        help='upsample policy for VM res')
    parser.add_argument('--hard_cap_at_e2u', action='store_true',
                        help='hard cap model size at even-to-uniform stage')
    
    # —— Selective-even split —— 
    parser.add_argument('--strict_even_kick', type=int, default=1500, 
                        help='first strict-even split (patch-level split) iteration')
    parser.add_argument('--strict_even_warmup_iters', type=int, default=300,  
                        help='lr warmup steps right after strict-even split')
    parser.add_argument('--strict_even_target_G', type=int, nargs='*', default=[3, 3, 3], 
                        help='grid G after first strict-even split')
    parser.add_argument('--split_even_kicks', type=int, nargs='*', 
                        default=[2200, 3600, 8000, 15000, 22000, 30000, 38000],
                        help='schedule of subsequent selective-even split events')
    parser.add_argument('--split_even_target_G', type=int, nargs='*', default=[3, 3, 3], 
                        help='target G for the even-split events')
    parser.add_argument('--even_gate_miss_q', type=float, default=None,
                        help='coverage gate for selective-even; require recent_missing_ratio_ema < this threshold')

    # —— VM-upsample —— 
    parser.add_argument('--vm_upsamp_list', type=int, nargs='*', default=[1800, 5200, 12000], 
                        help='schedule of VM upsample events')
    parser.add_argument('--perpatch_ups_topk_ratio', type=float, default=0.15,
                        help='fraction of eligible patches (by alpha_mass) to upsample this event')
    parser.add_argument('--perpatch_ups_topk', type=int, default=None,
                        help='fixed top-K patches to upsample; essentially use --perpatch_ups_topk_ratio instead')
    parser.add_argument('--perpatch_ups_min_k', type=int, default=6,
                        help='lower bound on number of patches to upsample per event')
    parser.add_argument('--perpatch_ups_max_k', type=int, default=24,
                        help='upper bound on number of patches to upsample per event')
    parser.add_argument('--ups_cooldown_after_split', type=int, default=800,
                        help='cooldown iters after a selective-even split before allowing VM upsample')
    parser.add_argument('--ups_gate_miss_q', type=float, default=0.08,
                        help='coverage gate for VM upsample; require recent_missing_ratio_ema < this threshold')
    parser.add_argument('--ups_gate_min_patch_factor', type=float, default=2.0,
                        help='require #patches >= prod(strict_even_target_G) * factor before VM upsample')
    parser.add_argument('--lr_upsample_reset', type=int, default=0,   
                        help='reset LR upon upsample (0/1)')

    # —— Post-uneven split ——
    parser.add_argument('--postcrit_apply', action='store_true', default=False,
                        help='enable the post-VM split phase (u+offset) after VM upsample')
    parser.add_argument('--postcrit_start_iter', type=int, default=10**9,
                        help='lower bound for post-VM split; ignored if --postcrit_apply=false')
    parser.add_argument('--split_after_upsamp_offset', type=int, default=0,
                        help='offset (iters) after each vm_upsamp_list step to run post-VM split if enabled')
    parser.add_argument('--postcrit_warmup_iters', type=int, default=300,
                        help='lr warmup steps right after an accepted restructure event (e.g., split or VM upsample)')
    parser.add_argument('--postcrit_warmup_floor', type=float, default=0.3,
                        help='starting LR factor during post-crit warmup (ramps from floor to 1.0)')
    parser.add_argument('--postcrit_gate_miss_q', type=float, default=0.05,
                        help='coverage gate for post-VM split/autoscale; require recent_missing_ratio_ema < this threshold')
    parser.add_argument('--postcrit_plateau_steps', type=int, default=800,
                        help='minimum steps window to judge late-stage plateau before post-VM ops')
    parser.add_argument('--postcrit_plateau_eps', type=float, default=0.05,
                        help='mean |ΔPSNR| per step below this means plateaued (late-stage only)')
    parser.add_argument('--postcrit_allow_up', action='store_true', default=False,
                        help='allow autoscale to increase ranks during post-VM only (usually keep False; can be True in probation)')
    
    # === Post-split field knowledge distillation (KD) ===  
    parser.add_argument('--post_event_kd', type=int, default=0,
                        help='enable short-horizon field KD after split (0/1)')
    parser.add_argument('--post_event_kd_every', type=int, default=2,
                        help='apply field KD every N iterations after a split (1 = every iter)')
    parser.add_argument('--post_event_kd_w', type=float, default=0.10,
                        help='global weight for field KD loss (applied to sigma/app MSE)')
    parser.add_argument('--post_event_kd_horizon', type=int, default=600,
                        help='how many iterations KD buffers remain active after a split')
    parser.add_argument('--post_event_kd_pts', type=int, default=2048,
                        help='max KD sample points drawn per iteration across active buffers')
    parser.add_argument('--kd_sigma_weight', type=float, default=1.0,
                        help='relative weight for density-field MSE inside KD')
    parser.add_argument('--kd_app_weight', type=float, default=1.0,
                        help='relative weight for appearance-field MSE inside KD')
    parser.add_argument('--kd_pts_per_child', type=int, default=256,
                        help='#KD samples per child cell when building teacher buffers')

    # === Seam tying / boundary agreement ===
    parser.add_argument('--seam_tying_enable', type=int, default=1,
                        help='enable seam handling: 0=off, 1=on')
    parser.add_argument('--seam_tying_mode', type=str, default='soft',
                        choices=['soft','hard'],
                        help='soft=loss on boundary samples; hard=post-step parameter tying')
    parser.add_argument('--seam_loss_w', type=float, default=0.05,
                        help='weight for seam consistency loss (when seam_tying_mode=soft)')
    parser.add_argument('--seam_sample_per_face', type=int, default=512,
                        help='#samples per adjacent-face for seam loss')
    parser.add_argument('--seam_eps', type=float, default=1e-3,
                        help='tiny offset to evaluate both sides of a boundary')

    # === Seam low-rank sharing ===
    parser.add_argument('--seam_lowrank_enable', action='store_true',
                        help='enable low-rank shared U + two-side coefficients (Ba/Bb) on patch seams')
    parser.add_argument('--seam_lowrank_scope', type=str, default='both',
                        choices=['plane','line','both'],
                        help='apply low-rank sharing to planes and/or lines')
    parser.add_argument('--seam_rank_sigma', type=int, default=8,
                        help='rank k for density seam banks')
    parser.add_argument('--seam_rank_app', type=int, default=8,
                        help='rank k for appearance seam banks')

    # === Split schedule warmup ===
    parser.add_argument('--split_warmup_iters', type=int, default=15000,
                        help='disable split events before this iter (only-UPS first)')

    # === Shared basis + per-patch mixing ===
    parser.add_argument('--basis_rank', type=int, default=16,
                        help='low-rank r for B@W (appearance only)')
    parser.add_argument('--basis_lowrank_enable', action='store_true',
                        help='use shared basis B (out=r->app_dim) + per-patch W (r x in_dim)')

    # === Rank governance ===
    parser.add_argument('--rank_auto_every', type=int, default=4000,
                        help='periodic rank prune/regrow interval (iters, <=0 disable)')
    parser.add_argument('--rank_keep_ratio', type=float, default=0.75,
                        help='keep top-L2 rows of W by this ratio when pruning')
    parser.add_argument('--rank_regrow', type=int, default=0,
                        help='rows to add back if KD/seam residual is high (0=off)')

    # === EMA teacher + boundary-weighted KD ===
    parser.add_argument('--kd_ema_enable', action='store_true',
                        help='use EMA teacher for KD')
    parser.add_argument('--kd_ema_decay', type=float, default=0.999,
                        help='EMA decay for teacher params')
    parser.add_argument('--kd_seam_boost', type=float, default=0.5,
                        help='fraction of KD samples drawn near seams (0~1)')
    parser.add_argument('--kd_seam_eps', type=float, default=1e-3,
                        help='offset for +/−eps seam sampling')

    # —— Split learning-rate shaping —— 
    parser.add_argument('--split_lr_pow', type=float, default=0.7)
    parser.add_argument('--split_lr_min', type=float, default=0.12)
    parser.add_argument('--split_cooldown', type=int, default=1000)
    parser.add_argument('--split_psnr_drop_thres', type=float, default=0.3, 
                        help='trigger split if PSNR drops more than this')
    
    # —— Restructure probation / Rollback —— 
    parser.add_argument('--restruct_immediate_abort_dB', type=float, default=4.0,
                        help='immediate rollback if ΔPSNR < -X dB right after restructure')
    parser.add_argument('--restruct_probation_allow_dB', type=float, default=3.0,
                        help='within probation, allow temporary ΔPSNR down to -X dB')
    parser.add_argument('--restruct_probation_iters', type=int, default=800,
                        help='delayed-judge window (iters) after restructure')
    parser.add_argument('--restruct_probation_final_tol_dB', type=float, default=0.5,
                        help='commit if current PSNR within X dB of pre-PSNR at end of probation')
    parser.add_argument('--restruct_probation_check_every', type=int, default=100,
                        help='how often to re-check PSNR during probation')

    # ===== Soft prune =====
    parser.add_argument('--softprune_start_iter', type=int, default=12000,
                        help='start iter for soft prune')
    parser.add_argument('--softprune_cooldown', type=int, default=1500,
                        help='cooldown (iters) between prunes')
    parser.add_argument('--softprune_alpha_q', type=float, default=0.90,
                        help='alpha quantile for pruning threshold')
    parser.add_argument('--softprune_alpha_min', type=float, default=5e-3,
                        help='absolute min alpha to keep')
    parser.add_argument('--softprune_min_reso', type=int, default=16,
                        help='min VM res to consider for prune')
    parser.add_argument('--softprune_keep_topk', type=int, default=1,
                        help='keep top-k patches even if low alpha')

    # ===== Export / Visualization =====
    parser.add_argument('--render_interval', type=int, default=5000,
                        help='interval for evaluation rendering')
    parser.add_argument('--vis_every', type=int, default=5000)
    parser.add_argument('--viz_patch_on_events', type=_parse_bool, default=True,
                        help='event viz (0/1/true/false)')
    parser.add_argument('--viz_patch_every', type=int, default=0,
                        help='periodic patch viz (0=disable)')
    parser.add_argument('--viz_dpi', type=int, default=110)
    parser.add_argument('--render_only', type=int, default=0)
    parser.add_argument('--render_train',  type=int, default=0,
                        help='render training views')
    parser.add_argument('--render_test', type=int, default=1,
                        help='render test views')
    parser.add_argument('--render_path', type=int, default=0)
    parser.add_argument('--export_mesh', type=int, default=0)
    parser.add_argument('--eval_all', action='store_true',
                        help='eval all checkpoints')

    # ===== Peak VRAM logging (CUDA) =====
    parser.add_argument('--peak_vram_on_start_reset', type=_parse_bool, default=True,
                        help='call torch.cuda.reset_peak_memory_stats() once before training loop if True')
    parser.add_argument('--peak_vram_log_every', type=int, default=0,
                        help='if > 0, log CUDA peak memory every N iterations; if 0, fallback to periodic heartbeat')
    parser.add_argument('--peak_vram_tag_prefix', type=str, default='mem',
                        help='TensorBoard tag prefix for peak VRAM metrics')

    # ===== Debug / Misc =====
    parser.add_argument('--ckpt', type=str, default='',
                        help='load checkpoint from path (optional)')
    parser.add_argument('--alpha_mass_n', type=int, default=1024,
                        help='number of random alpha mass samples per patch')
    parser.add_argument('--sdf', action='store_true')
    parser.add_argument('--camera_sweep', action='store_true')
    parser.add_argument('--camera_sweep_step', type=float, default=0.0)
    parser.add_argument('--fovy_deg', type=float, default=0.0)
    parser.add_argument('--N_vis', type=int, default=5)
    parser.add_argument('--Vis_stride', type=int, default=1)
    parser.add_argument('--grad_scale_basis', type=float, default=1.0)
    parser.add_argument('--basis_type', type=str, default='sh')
    parser.add_argument('--update_basis_min', type=int, default=0)
    parser.add_argument('--update_basis_every', type=int, default=0)
    parser.add_argument('--update_basis_topk', type=int, default=0)
    parser.add_argument('--weights_regularize', type=float, default=0.0)
    parser.add_argument('--rm_weight', type=float, default=0.0)
    parser.add_argument('--rm_thre', type=float, default=0.0)
    parser.add_argument('--reg_debug_once', action='store_true',
                        help='run regularization schedule debug and exit')
    parser.add_argument('--reg_debug_iters32', type=int, default=300,
                        help='iters with min_res=32 in reg debug')
    parser.add_argument('--reg_debug_total', type=int, default=700,
                        help='total iters in reg debug')
    parser.add_argument('--n_test', type=int, default=10,
                        help='number of test frames to render')
    parser.add_argument('--no_lr_decay', action='store_true',
                        help='disable lr decay (debug)')
    parser.add_argument('--warn_interval', type=int, default=200,
                        help='debug warn interval (iters)')
    parser.add_argument('--warn_min_rays', type=int, default=512,
                        help='debug min rays to warn on')
    parser.add_argument('--debug_map_stats', action='store_true',
                        help='print map stats for debugging')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
