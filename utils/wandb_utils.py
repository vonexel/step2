import wandb
import torch

"""
The module serves as an abstraction layer between core training code and W&B's API, promoting code cleanliness while maintaining comprehensive experiment tracking capabilities.
It facilitates reproducibility by systematically recording hyperparameters, model architecture details, and training progression metrics.
"""


def wandb_init(args, model, project="step2"):
    wandb.init(
        project=project, name=args.exp_name, config=vars(args), tags=[args.dataname]
    )
    # Trainning parameters logging
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params})


def wandb_log(metrics, step=None):
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def wandb_save_checkpoint(model, optimizer, step, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )
    artifact = wandb.Artifact(f"checkpoint_{wandb.run.name}_{step}", type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
