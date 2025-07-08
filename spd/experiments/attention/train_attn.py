"""Train single-head attention model for SPD decomposition."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt
from tqdm import tqdm, trange
import numpy as np
import torch.utils.data

from spd.experiments.attention.models import SingleHeadAttentionModel  # To be created
from spd.log import logger
from spd.utils import set_seed
from spd.experiments.attention.models import SingleHeadAttentionModel, AttnModelConfig
from spd.data_utils import DatasetGeneratedDataLoader, SkipTrigramDataset
wandb.require("core")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"{name}: NO GRADIENT")

class AttnTrainConfig(BaseModel):
    """Configuration for training the attention model."""
    wandb_entity: str | None = None
    wandb_project: str | None = None
    attention_model_config: AttnModelConfig
    batch_size: PositiveInt = 32
    steps: PositiveInt = 5000
    seed: int = 0
    lr: float = 1e-3
    lr_schedule: Literal["linear", "cosine", "constant"] = "constant"
    # Data generation parameters
    data_type: Literal["random", "copying", "next_token"] = "random"
    n_trigrams: PositiveInt = 32  # Number of trigrams to generate for the dataset

def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: SingleHeadAttentionModel,
    dataloader,  # TODO: Type this properly when we have data_utils
    log_wandb: bool,
    steps: int,
    print_freq: int,
    lr: float,
    lr_schedule: Literal["linear", "cosine", "constant"],
) -> None:
    """Train the attention model on language modeling task."""
    
    if lr_schedule == "linear":
        lr_schedule_fn = linear_lr  
    elif lr_schedule == "cosine":
        lr_schedule_fn = cosine_decay_lr
    elif lr_schedule == "constant":
        lr_schedule_fn = constant_lr
    else:
        raise ValueError(f"Invalid lr_schedule: {lr_schedule}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule_fn(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
                
            opt.zero_grad(set_to_none=True)
            
            # Get batch with targets for skip-trigram task
            sequences, targets = dataloader.dataset.generate_batch_with_targets(dataloader.batch_size)
            
            # Forward pass
            logits = model(sequences)  # [batch, seq, vocab_size]
            
            # Skip-trigram loss: only predict the target token after the final position
            # We want to predict what comes after the trigger2 (which is at position -1)
            final_logits = logits[:, -1, :]  # [batch, vocab_size] - logits for final position
            
            loss = torch.nn.functional.cross_entropy(
                final_logits,  # [batch, vocab_size]
                targets,       # [batch] - the target tokens
            )
            
            loss.backward()
            opt.step()
            if step % 100 == 0:  # Check every 100 steps
                check_gradients(model)
            # Add this debugging code in your train function after getting sequences and targets
            if step % 100 == 0:
                print(f"\nStep {step} debugging:")
                print(f"Sequences shape: {sequences.shape}")
                print(f"Targets shape: {targets.shape}")
                print(f"Sample sequence: {sequences[0].cpu().numpy()}")
                print(f"Sample target: {targets[0].item()}")
                print(f"Last token in sequence: {sequences[0, -1].item()}")
                print(f"Logits shape: {logits.shape}")
                print(f"Final logits shape: {final_logits.shape}")
                print(f"Loss: {loss.item()}")
                
                # Check if we're actually predicting the right thing
                predicted_token = final_logits[0].argmax().item()
                print(f"Predicted token: {predicted_token}, Target: {targets[0].item()}")
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                t.set_postfix(loss=loss.item(), lr=step_lr)
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)
                    
                    
# Then replace the get_model_and_dataloader function:
def get_model_and_dataloader(
    config: AttnTrainConfig, device: str
) -> tuple[SingleHeadAttentionModel, DatasetGeneratedDataLoader[torch.Tensor]]:
    """Create model and dataloader for attention training."""
    
    model = SingleHeadAttentionModel(config.attention_model_config)
    model.to(device)
    
    dataset = SkipTrigramDataset(
        vocab_size=config.attention_model_config.vocab_size,
        seq_len=config.attention_model_config.seq_len,
        device=device,
    )
    
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    return model, dataloader


def run_train(config: AttnTrainConfig, device: str) -> None:
    """Main training runner for attention model."""
    
    model, dataloader = get_model_and_dataloader(config, device)
    
    # Create run name
    model_cfg = config.attention_model_config
    run_name = (
        f"attention_vocab{model_cfg.vocab_size}_d{model_cfg.d_model}_"
        f"seq{model_cfg.seq_len}_data{config.data_type}_seed{config.seed}"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "toy_out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.wandb_project:
        wandb.init(entity = config.wandb_entity, project=config.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "attention_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    # Train model
    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
        print_freq=100,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
    )

    # Save trained model
    model_path = out_dir / "attention_model.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # TODO: Add attention-specific analysis
    # - Visualize attention patterns
    # - Test on specific sequences
    # - Analyze what the model learned

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AttnTrainConfig(
        wandb_project="spd-train-attn",
        attn_model_config=AttnModelConfig(
            vocab_size=20,
            d_model = 10,
            seq_len=8
        ),

        batch_size=4096,
        steps=10000,
        seed=0,
        lr=5e-3,
        lr_schedule="constant",
        # synced_inputs=[[5, 6], [0, 2, 3]],
    )

    set_seed(config.seed)

    run_train(config, device)
