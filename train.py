import os
import tqdm
import torch
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import datetime
from omegaconf import OmegaConf
from einops import rearrange, repeat

from utils.utils import instantiate_from_config, load_model_checkpoint, get_cosine_schedule_with_warmup, AverageMeter
from utils.video_dataset import sequence_collate, RepeatedDataset, ShowHowToDataset

from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.openaimodel3d import  ResBlock, TemporalConvBlock, Downsample, Upsample

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def _set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def _apply_ft_mode(unet: torch.nn.Module, mode: str):
    """
    mode:
      - 'freeze_spatial': 空間系（SpatialTransformer / ResBlockの2D conv群 / Downsample / Upsample / out conv）を凍結。
                          それ以外（Temporal 系や time_embed 等）は学習。
      - 'all': 何もしない。
    """

    if mode == "all":
        # UNet params: trainable=1,413,284,420 / total=1,413,284,420
        return

    elif mode == "freeze_spatial":
        # UNet params: trainable=722,651,018 / total=1,536,134,858 (improve & c_aware)
        # UNet params: trainable=599,800,580 / total=1,413,284,420 (fs)
        # UNet params: trainable=549,424,256 / total=1,413,284,420 (fs noattn)
        
        # いったん UNet 全体を学習可に
        _set_requires_grad(unet, True)

         # in と out を丸ごと凍結
        _set_requires_grad(unet.input_blocks[0], False)  # 最初の conv（in）
        _set_requires_grad(unet.out, False)              # 末尾の norm+SiLU+zero(conv)（out）

        # SpatialTransformer / Downsample / Upsample / ResBlock を凍結
        for m in unet.modules():
            
            if isinstance(m, SpatialTransformer):
                _set_requires_grad(m, False)

                # ただしcross attention は学習可に戻す
                if hasattr(m, "transformer_blocks"): # nn.ModuleList([BasicTransformerBlock(..)])
                    for block in m.transformer_blocks: # BasicTransformerBlock

                        if hasattr(block, "attn2"):
                            _set_requires_grad(block.attn2, True)
                        
            if isinstance(m, (Downsample, Upsample)):
                _set_requires_grad(m, False)

            if isinstance(m, ResBlock):
                # ResBlock 丸ごと凍結
                _set_requires_grad(m, False)
                # ただし temporal conv は学習可に戻す（temopral_conv はミスではなく正しい変数名）
                if getattr(m, "use_temporal_conv", False) and hasattr(m, "temopral_conv"):
                    _set_requires_grad(m.temopral_conv, True)
    
    else:
        raise ValueError(f"unknown ft_mode: {mode}")
        

# -----------------------------------------------------------
# Main training function
# -----------------------------------------------------------

def main(args):
   # ---- replace the env detection block in main(args) ----
    def _get_env_int(keys, default):
        for k in keys:
            v = os.environ.get(k)
            if v is not None and str(v).isdigit():
                return int(v)
        return default

    ngpus_per_node = torch.cuda.device_count()  # (=1想定)
    node_count = _get_env_int(["OMPI_COMM_WORLD_SIZE", "PBS_NP", "SLURM_NPROCS"], 1)
    node_rank  = _get_env_int(["OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"], 0)

    # ★ PBSのJOBID or 共有ファイルを優先的に使う（ランダム生成をやめる）
    job_id = os.environ.get("PBS_JOBID") or os.environ.get("OMPI_COMM_WORLD_JOBID") or "0"

    # ★ PBSジョブで作った共有ファイルをそのまま使う（全ランクで同一パス）
    dist_file_env = os.environ.get("DIST_FILE")
    if dist_file_env is not None:
        dist_url = f"file://{dist_file_env}"
    else:
        # フォールバック（共有FS上に置く）
        workdir = os.environ.get("PBS_O_WORKDIR", os.getcwd())
        dist_url = f"file://{os.path.join(workdir, 'distfile.' + str(job_id))}"

    print(f"[rank?] node_count={node_count} ngpus_per_node={ngpus_per_node} node_rank={node_rank} dist_url={dist_url}", flush=True)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=({
        "ngpus_per_node": ngpus_per_node,
        "node_count": node_count,
        "node_rank": node_rank,
        "dist_url": dist_url,
        "job_id": job_id
    }, args))


def main_worker(local_rank, cluster_args, args):
    print(f"Hi from local rank {local_rank}!", flush=True)

    # ----------------------------
    # configure distributed training
    # ----------------------------
    world_size = cluster_args["node_count"] * cluster_args["ngpus_per_node"]
    global_rank = cluster_args["node_rank"] * cluster_args["ngpus_per_node"] + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=cluster_args["dist_url"],
        world_size=world_size,
        rank=global_rank,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        store_dir = f"./logs/train/{args.exp_name}/" + datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            print(f"# {k}: {v}")
        print(f"# store_dir: {store_dir}")
        print(f"# effective_batch_size: {world_size * args.local_batch_size}", flush=True)


    ###############
    # DATASET
    ###############
    n_epochs = 200
    save_every_n_epochs = 1

    train_ds = []
    for i in range(2, args.max_seq_len + 1):
        train_ds.append(RepeatedDataset(
            ShowHowToDataset(args.dataset_root, video_length=i), epoch_len=16000))

    train_samplers = [None for _ in train_ds]
    if world_size > 1:
        train_samplers = [torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, drop_last=True) for ds in train_ds]

    train_ds_iters = [torch.utils.data.DataLoader(
        ds, batch_size=args.local_batch_size, shuffle=world_size == 1, drop_last=True, num_workers=1, 
        pin_memory=True, sampler=train_sampler, collate_fn=sequence_collate) for ds, train_sampler in zip(train_ds, train_samplers)]

    ###############
    # MODEL
    ###############
    learning_rate = 2e-5

    config = OmegaConf.load("./configs/inference_t2v_512_v2.0.yaml")["model"]
    if args.use_improve_contextualizer:
        print("<config> Using improved contextualizer", flush=True)
        config['params']['unet_config']['params']['improve_contextualizer'] = True
    elif args.use_contextualizer:
        print("<config> Using contextualizer", flush=True)
        config['params']['unet_config']['params']['contextualizer'] = True
    if args.use_c_aware:
        print("<config> Using Context-Aware Temporal Attention", flush=True)
        config['params']['unet_config']['params']['c_aware'] = True
    model = instantiate_from_config(config)

    if args.pretrained_ckpt is None:
        model = load_model_checkpoint(model, args.ckpt_path)
        print(f"Loaded model from {args.ckpt_path}", flush=True)
    else:
        state_dict = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model from {args.pretrained_ckpt}", flush=True)

    model.to(device)

    # UNet は model.diffusion_model
    unet = model.model.diffusion_model
    _apply_ft_mode(unet, args.ft_mode)

    # 学習させる param だけを Optimizer に渡す
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if global_rank == 0:
        total = sum(p.numel() for p in unet.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(f"# FT mode: {args.ft_mode}", flush=True)
        print(f"# UNet params: trainable={trainable:,} / total={total:,}", flush=True)

    model_parallel = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )
    print(f"Model distributed to gpu {global_rank}!", flush=True)

    ###############
    # OPTIMIZER
    ###############
    parameters2train = trainable_params # model_parallel.module.model.parameters()
    optim = torch.optim.AdamW(parameters2train, lr=learning_rate)
    if args.pretrained_ckpt is None:
        past_epochs = 0
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            len(train_ds_iters[0]),
            len(train_ds_iters[0]) * n_epochs
        )
    else:
        past_epochs = int(args.pretrained_ckpt.split("_")[-1].split(".")[0])
        steps_done = past_epochs * len(train_ds_iters[0])
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=0,
            num_training_steps=len(train_ds_iters[0]) * n_epochs
        )
        scheduler.step(steps_done)
        print(f"Resumed from epoch {past_epochs} / step {steps_done}", flush=True)
    loss_metric = AverageMeter()

    for epoch in range(past_epochs + 1, n_epochs + 1):
        if world_size > 1:
            for train_sampler in train_samplers:
                train_sampler.set_epoch(epoch)

        iterator = tqdm.tqdm(train_ds_iters[-1]) if global_rank == 0 else train_ds_iters[-1]
        other_iterators = [iter(ds) for ds in train_ds_iters[:-1]]
        for video_frames, prompts in iterator:
            # gather data for all lengths
            iterator_data = [(video_frames, prompts)]
            for other_iterator in other_iterators:
                iterator_data.append(next(other_iterator))

            for video_frames, prompts in iterator_data:
                B, C, T, H, W = video_frames.shape
                frame_stride = torch.ones((B,), dtype=torch.long, device=device)

                with torch.no_grad():
                    text_emb = model.get_learned_conditioning(prompts)
                    z = get_latent_z(model, video_frames.to(device))
                
                # 10% dropout 
                if random.random() < 0.1:
                    text_emb = model.get_learned_conditioning(len(prompts) * [""])

                tB, tL, tC = text_emb.shape
                cond = {
                    "c_crossattn": [text_emb],
                }

                t = torch.randint(0, model.num_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z)
                x_noisy = model.q_sample(x_start=z, t=t, noise=noise)

                model_output = model_parallel(x_noisy, t, cond, fps=frame_stride)

                loss = torch.nn.functional.mse_loss(noise, model_output, reduction='none')
                loss = loss.mean([1, 2, 3, 4])
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()  # DistributedDataParallel does gradient averaging, i.e. loss is x-times smaller when trained on more GPUs
                optim.step()
                loss_metric.update(loss.item())

            scheduler.step()

        if global_rank == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_metric.value:.4f}", flush=True)

            os.makedirs(store_dir, exist_ok=True)
            with open(os.path.join(store_dir, "losses.txt"), "a") as f:
                f.write(f"{epoch:03d}{loss_metric.value:12.4f}\n")
            loss_metric.reset()

            if epoch % save_every_n_epochs == 0:
                torch.save(model_parallel.module.state_dict(), os.path.join(store_dir, f"model_{epoch:03d}.pt"))

    print("Training finished.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--ckpt_path", type=str, default="./weights/model.ckpt") # if args.pretrained_ckpt is None, use this path
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--pretrained_ckpt", type=str, default=None) # resume training from this checkpoint
    parser.add_argument("--exp_name", type=str, required=True)
    # full, freeze_spatial, freeze_spatial_contextualizer, freeze_spatial_improve, freeze_spatial_improve_caware

    parser.add_argument(
        "--ft_mode",
        type=str,
        default="all",
        choices=["all", "freeze_spatial"],
    )
    parser.add_argument("--use_contextualizer", action="store_true")
    parser.add_argument("--use_improve_contextualizer", action="store_true")
    parser.add_argument("--use_c_aware", action="store_true")

    main(parser.parse_args())
