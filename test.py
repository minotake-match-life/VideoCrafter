import os
import time

import torch
import argparse
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange, repeat

from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config


@torch.no_grad()
def synthesis(model, prompts, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.,
              unconditional_guidance_scale=1.0, **kwargs):
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fps = torch.tensor([1.] * batch_size, dtype=torch.long, device=model.device)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [cond_emb]}
    
    if unconditional_guidance_scale != 1.0:
        uc_emb = model.get_learned_conditioning(batch_size * [""])
        uc = {
            "c_crossattn": [uc_emb]
        }
    else:
        uc = None

    z0 = None
    cond_mask = None
    x_T = None
    timesteps = None

    batch_variants = []
    for _ in range(n_samples):
        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=noise_shape[1:],
                                             verbose=True,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             mask=cond_mask,
                                             x0=cond_z0,
                                             fps=fps,
                                             x_T=x_T,
                                             timesteps=timesteps,
                                             **kwargs)
        # reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    # variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def main(args):
    print("Loading model...")
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
    model = model.cuda()

    # load U-Net and VAE weights
    print("load model from", args.ckpt_path)
    state_dict = torch.load(args.ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # prepare dataset
    print("Loading dataset...")
    with open(args.prompt_file, "r") as f:
        dataset = []
        for line in f.readlines():
            if line.strip() == "" or line.startswith("#"):
                continue
            image_path, prompts = line.split(args.delimiter, 1)
            prompts = [p.strip().strip('"') for p in prompts.split(args.delimiter)]

            dataset.append((image_path.strip(), prompts))
    
    for idx, (_, prompts) in enumerate(dataset):
        print(f"Processing {idx + 1}/{len(dataset)}", flush=True)

        n_frames = len(prompts)
        noise_shape = [1, 4, n_frames, 32, 32]  # B, C, T, H, W

        samples = synthesis(
            model, prompts, noise_shape, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
            unconditional_guidance_scale=args.unconditional_guidance_scale
        )

        # n_samples=1, B=1, C, T, h, w
        samples = samples.squeeze(0).squeeze(0)
        samples = samples.clamp_(-1, 1).add_(1.).mul_(255 / 2)
        samples = samples.to(torch.uint8).permute(1, 2, 3, 0)
        samples = samples.cpu().numpy()

        output_image = Image.fromarray(np.concatenate(samples, axis=1))
        os.makedirs(args.output_dir, exist_ok=True)
        output_image.save(os.path.join(args.output_dir, f"seq_{idx}.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default="/work/xg25g011/x10574/ShowHowData/data/ShowHowToTest/prompt_file.txt",
                        help="text file with image paths and prompts")
    parser.add_argument("--delimiter", type=str, default="|", help="delimiter for image paths and prompts")
    parser.add_argument('--output_dir', type=str, default=f"./output/{time.strftime('%Y%m%d_%H%M')}", help='Output directory')

    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12, help="prompt classifier-free guidance")

    parser.add_argument("--use_contextualizer", action="store_true")
    parser.add_argument("--use_improve_contextualizer", action="store_true")
    parser.add_argument("--use_c_aware", action="store_true")


    args = parser.parse_args()

    main(args)
