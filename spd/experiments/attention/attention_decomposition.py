"""Run spd on a Attn model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
import yaml

from spd.configs import Config
from spd.data_utils import DatasetGeneratedDataLoader, SkipTrigramDataset
from spd.experiments.attention.models import SingleHeadAttentionModel, AttnModelConfig
from spd.log import logger
from spd.plotting import create_toy_model_plot_results
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, attn_model_config: AttnModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"vocab{attn_model_config.vocab_size}_"    # Changed from n_features
        run_suffix += f"d{attn_model_config.d_model}_"          # Changed from n_hidden
        run_suffix += f"seq{attn_model_config.seq_len}"         # Changed from n_hidden_layers
    return config.wandb_run_name_prefix + run_suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    attn_model: SingleHeadAttentionModel,
    attn_model_train_config_dict: dict[str, Any],
) -> None:
    torch.save(attn_model.state_dict(), out_dir / "attn.pth")

    with open(out_dir / "attn_config.yaml", "w") as f:
        yaml.dump(attn_model_train_config_dict, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "attn.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "attn_config.yaml"), base_path=out_dir, policy="now")


def main(config_path_or_obj: Path | str | Config) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, project=config.wandb_project)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    
    # Manual loading instead of from_pretrained
    attn_model_config = AttnModelConfig(
        vocab_size=20,  # Make sure these match your trained model
        d_model=16,  # Make sure these match your trained model
        seq_len=8,
    )
    
    target_model = SingleHeadAttentionModel(attn_model_config)
    # Add this right before the torch.load call
    print(f"Trying to load model from: {config.pretrained_model_path}")
    print(f"File exists: {Path(config.pretrained_model_path).exists()}")
    if Path(config.pretrained_model_path).exists():
            print(f"File size: {Path(config.pretrained_model_path).stat().st_size} bytes")
            # Check first few bytes
            with open(config.pretrained_model_path, 'rb') as f:
                first_bytes = f.read(10)
                print(f"First 10 bytes: {first_bytes}")
    target_model.load_state_dict(torch.load(config.pretrained_model_path, map_location=device, weights_only=False))    
    target_model = target_model.to(device)
    target_model.eval()

    target_model_train_config_dict = {
        "vocab_size": attn_model_config.vocab_size,
        "d_model": attn_model_config.d_model,
        "seq_len": attn_model_config.seq_len,
    }

    run_name = get_run_name(config=config, attn_model_config=attn_model_config)  # Use attn_model_config
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "TEST_OUT" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        attn_model=target_model,
        attn_model_train_config_dict=target_model_train_config_dict,
    )

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = SkipTrigramDataset(
        vocab_size=attn_model_config.vocab_size,  # Use attn_model_config
        seq_len=attn_model_config.seq_len,        # Use attn_model_config
        device=device
    )
    
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        plot_results_fn=None,
        tied_weights=None,  # Changed from False to None
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
