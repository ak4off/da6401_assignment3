import wandb


def setup_wandb(project_name, entity_name, model, run_name=None):
    wandb.init(
        project=project_name,
        entity=entity_name,
        name=run_name,  # Custom run name
        settings=wandb.Settings(init_timeout=300)  # ← Increase timeout here
    )
    wandb.watch(model, log="all", log_freq=100)


def log_metrics(metrics, epoch, step):
    wandb.log({**metrics, "epoch": epoch, "step": step})


