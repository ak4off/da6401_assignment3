import wandb

# def setup_wandb(project_name, entity_name, model):
#     wandb.init(project=project_name, entity=entity_name)
#     wandb.watch(model, log="all", log_freq=100)
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

# def log_model(model, epoch):
#     wandb.watch(model, log="all", log_freq=100)
#     wandb.save(f"model_epoch_{epoch}.pth")
# def log_config(config):
#     wandb.config.update(config)
# def finish_wandb():
#     wandb.finish()  
# def log_hyperparameters(hyperparameters):
#     wandb.config.update(hyperparameters)
# def log_artifact(artifact_name, artifact_type, artifact_description):
#     artifact = wandb.Artifact(artifact_name, type=artifact_type, description=artifact_description)
#     wandb.log_artifact(artifact)
# def log_table(table_name, data):
#     table = wandb.Table(data=data)
#     wandb.log({table_name: table})
