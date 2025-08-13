# Usual misc.
import os
from collections import defaultdict
import random
from pathlib import Path
import pickle
from tqdm import tqdm
import json

# Computation
import numpy as np
import torch
import torch.nn.functional as F
import einops

# Config
import hydra

# Data & Model
from src.data import make_datasets
from omegaconf import OmegaConf

# Logging
import wandb


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ce_from_probs(preds, target, reduction="mean"):
    # Preds : B V N
    # Target : B N
    preds[preds == 0] = 1e-8  # Avoid -inf from log when p=0

    target = target.clone()  # Avoid in-place modifications, can be re-used
    target[preds.sum(dim=1).isnan()] = -100  # Ignore degenerate cases
    return F.nll_loss(torch.log(preds), target, reduction=reduction)


def update_metrics(
    model_loss, cut_model_loss, empirical_loss, oracle_loss, metrics, split
):
    if model_loss is not None:
        metrics[split]["model_loss"].append(model_loss.item())
    metrics[split]["cut_model_loss"].append(cut_model_loss.item())
    metrics[split]["oracle_loss"].append(oracle_loss.item())
    metrics[split]["empirical_loss"].append(empirical_loss.item())
    metrics[split]["d_model_oracle"].append((cut_model_loss - oracle_loss).item())
    metrics[split]["d_model_empirical"].append((cut_model_loss - empirical_loss).item())


class CkptManager:
    ckpt_filename = "checkpoint.pkl"
    logs_filename = "logs.jsonl"

    def __init__(self, cfg):
        self.folder_name = cfg.logger.exp_id

        self.folder_path = Path("runs") / self.folder_name
        self.folder_path.mkdir(parents=True, exist_ok=False)

        # Dumpy hydra config to folder_path
        with open(self.folder_path / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

        self.ckpt_path = self.folder_path / self.ckpt_filename
        self.logs_path = self.folder_path / self.logs_filename

        self.use_wandb = cfg.logger.project_name is not None
        if cfg.logger.project_name:
            wandb.init(
                project=cfg.logger.project_name,
                name=cfg.logger.exp_id,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.folder_path),
            )

    def to_checkpoint(self, **ckpt_data):
        for k, v in ckpt_data.items():
            if hasattr(v, "state_dict"):
                ckpt_data[k] = v.state_dict()

        with open(self.ckpt_path, "wb") as f:
            pickle.dump(ckpt_data, f)

        with open(self.logs_path, "a") as f:
            f.write(json.dumps(ckpt_data["report"]) + "\n")

        if self.use_wandb:
            wandb.log({"step": ckpt_data.get("step", 0), **ckpt_data["report"]})


def train_markov(cfg):
    seed_everything(cfg.seed)

    # Data
    train_dl, valid_dl = make_datasets(cfg)

    # Model
    model = hydra.utils.instantiate(cfg.model).to(cfg.device, non_blocking=True)
    model.train()
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimization
    optim = hydra.utils.instantiate(
        cfg.optim, params=[p for p in model.parameters() if p.requires_grad]
    )
    cross_entropy = torch.nn.CrossEntropyLoss()

    metrics = {split: defaultdict(list) for split in ["train", "valid"]}

    ckpt_manager = CkptManager(cfg)

    step = 0
    for batch in tqdm(train_dl, desc="Training", leave=False):
        step += 1
        # # Forward pass
        walk = batch["walk"].to(cfg.device, non_blocking=True)
        target = batch["target"].to(cfg.device, non_blocking=True)
        cut_target = batch["cut_target"].to(cfg.device, non_blocking=True)
        oracle = batch["oracle"].to(cfg.device, non_blocking=True)
        empirical = batch["empirical"].to(cfg.device, non_blocking=True)

        # Len of last seq is WALK_LEN - 1 (because last node is for target)
        preds = model(walk).logits
        preds = einops.rearrange(preds, "b n v -> b v n")
        loss = cross_entropy(preds, target)

        loss.backward()
        if cfg.train.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
        optim.step()
        optim.zero_grad()

        with torch.no_grad():
            loss_model = cross_entropy(preds[:, :, -20:], cut_target)
            loss_empirical = ce_from_probs(
                einops.rearrange(empirical, "b n v -> b v n"), cut_target
            )

            loss_oracle = ce_from_probs(
                einops.rearrange(oracle, "b n v -> b v n"), cut_target
            )

            update_metrics(
                loss, loss_model, loss_empirical, loss_oracle, metrics, "train"
            )

        if step % cfg.train.val_every == 0:
            report = {
                "train/step": step,
                "train/n_params": model_n_params,
                "lr": optim.param_groups[0]["lr"],
                **{"train/" + k: sum(v) / len(v) for k, v in metrics["train"].items()},
            }

            with torch.no_grad():
                model.eval()
                for batch in tqdm(valid_dl, desc="Validating", leave=False):
                    # Forward pass
                    walk = batch["walk"].to(cfg.device, non_blocking=True)
                    target = batch["target"].to(cfg.device, non_blocking=True)
                    cut_target = batch["cut_target"].to(cfg.device, non_blocking=True)
                    oracle = batch["oracle"].to(cfg.device, non_blocking=True)
                    empirical = batch["empirical"].to(cfg.device, non_blocking=True)

                    preds = model(walk).logits
                    preds = einops.rearrange(preds, "b n v -> b v n")
                    loss = cross_entropy(preds, target)

                    loss_model = cross_entropy(preds[:, :, -20:], cut_target)
                    loss_empirical = ce_from_probs(
                        einops.rearrange(empirical, "b n v -> b v n"), cut_target
                    )
                    loss_oracle = ce_from_probs(
                        einops.rearrange(oracle, "b n v -> b v n"), cut_target
                    )

                    update_metrics(
                        loss, loss_model, loss_empirical, loss_oracle, metrics, "valid"
                    )

                report.update(
                    {"valid/" + k: sum(v) / len(v) for k, v in metrics["valid"].items()}
                )

            print("Validation report: ", report)
            ckpt_manager.to_checkpoint(
                model=model, optim=optim, step=step, cfg=cfg, report=report
            )

            # Reset the holder and training state.
            metrics = {split: defaultdict(list) for split in ["train", "valid"]}
            model.train()


@hydra.main(version_base="1.3", config_path="config", config_name="train.yaml")
def main(cfg):
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    OmegaConf.resolve(cfg)

    train_markov(cfg)


if __name__ == "__main__":
    main()
