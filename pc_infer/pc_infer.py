import torch
import os
import time
# from utils.logger import *
from utils.logger import get_logger, get_root_logger, print_log
import numpy as np
import open3d as o3d
from models.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e.diffusion.sampler import PointCloudSampler
from models.point_e.models.configs import MODEL_CONFIGS, model_from_config
import models.point_e.util.builder as builder
from tqdm.auto import tqdm
from extensions.chamfer_dist import ChamferDistanceL1_PM
from utils import misc
from datasets.io import IO
from utils import parser, dist_utils, misc
from utils.config import get_config

os.environ["OMP_NUM_THREADS"] = "4"


def pc_norm(pc):
    """pc: NxC, return NxC"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) * 2
    pc = pc / m
    return pc

def pc_norm_stats(pc):
    """pc: NxC, return NxC"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) * 2
    pc = pc / m
    return pc, centroid, m


def savePC(partial, samples, save_path):

    np.savetxt(
        os.path.join(save_path, "demo.txt"),
        samples[0].detach().cpu().numpy(),
        delimiter=";",
    )
    np.savetxt(
        os.path.join(save_path, "input.txt"),
        partial[0].detach().cpu().numpy(),
        delimiter=";",
    )



class ProtoCompInference:

    def __init__(self, proj_base: str=os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))):
        # temp fix for relative path
        cur_cwd = os.getcwd()
        os.chdir(proj_base)

        ckpt_path = f"{proj_base}/pcn.pth"
        config_path = f"{proj_base}/cfgs/PCN_models/ProtoComp.yaml"
        exp_name = "example"
        print(f"ckpt_path: {ckpt_path}")
        print(f"config_path: {config_path}")

        args = parser.get_args(f"--demo --ckpts {ckpt_path}  --config {config_path} --exp_name {exp_name}".split())
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_file = os.path.join(args.experiment_path, f"{timestamp}.log")
        logger = get_root_logger(log_file=log_file, name=args.log_name)

        config = get_config(args, logger=logger)

        assert args.ckpts is not None
        assert args.config is not None
        args.use_gpu = True

        logger = get_logger(args.log_name)
        print_log("Tester start ... ", logger=logger)
        base_model = model_from_config(
            MODEL_CONFIGS["base40M-textvec"], config.model, device=args.local_rank
        )

        builder.load_model(base_model, args.ckpts, logger=logger)
        if args.use_gpu:
            base_model.to(args.local_rank)

        device = args.local_rank
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["base40M-textvec"])
        sampler = PointCloudSampler(
            device=device,
            models=[base_model],
            diffusions=[base_diffusion],
            num_points=[1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0],
            karras_steps=[64],
            sigma_min=[1e-3],
            sigma_max=[120],
            s_churn=[3],
            use_karras=[True],
            model_kwargs_key_filter=["texts"],
        )
        base_model.eval()

        self.base_model = base_model
        self.args = args
        self.config = config
        self.sampler = sampler
        self.logger = logger

        os.chdir(cur_cwd)


    def demo(
        self,
        partial_pc: np.array,
        prompt_str: str,
        debug: bool = False,
    ):

        # data = IO.get(pc_path).astype(np.float32)
        data, centroid, scale = pc_norm_stats(data)
        data = torch.from_numpy(data).float()

        prompt = [prompt_str]

        partial = data.to(torch.float32).cuda().unsqueeze(0)

        partial = misc.fps(partial, 2048)

        for x in tqdm(
            self.sampler.sample_batch_progressive(
                partial=partial,
                batch_size=1,
                model_kwargs=dict(texts=prompt),
            )
        ):
            samples = x
            samples = samples[:, :3, :].transpose(1, 2)
        print("PROMPT: ", prompt)

        if debug:
            import viser

            server = viser.ViserServer(port=9082)
            base_frame = server.scene.add_frame(
                "/frames",
                # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
                position=(0, 0, 0),
                show_axes=True,
            )

            partial = partial[0].detach().cpu().numpy()
            samples = samples[0].detach().cpu().numpy()

            colors_partial = np.zeros_like(partial)
            colors_partial[:, 0] = 0.8
            server.scene.add_point_cloud(
                        "/frames/partial",
                        points=partial,
                        point_size=0.005,
                        colors=colors_partial,
                    )

            colors_complete = np.zeros_like(samples)
            colors_complete[:, 2] = 0.8
            server.scene.add_point_cloud(
                        "/frames/complete",
                        points=samples,
                        point_size=0.005,
                        colors=colors_complete,
                    )

            while True:
                time.sleep(0.01)

        complete_pc = samples[0].detach().cpu().numpy() * scale + centroid
        return complete_pc
