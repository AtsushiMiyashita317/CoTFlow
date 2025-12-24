
import os
import yaml
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import cotflow.datasets as datasets
from pytorch_lightning.callbacks import ModelCheckpoint

import cotflow.utils as utils
from cotflow.nets.cotflow import CoTGlowModule
from cotflow.nets.predictor import PredictorLightningModule

@hydra.main(config_path=os.path.abspath("config"), config_name='cotglow.yaml', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # hydra.run.dirを取得し、出力ディレクトリとして使う
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    utils.set_seed(cfg.seed)

    # データセット
    train_ds = getattr(datasets, cfg.train.dataset.name)(**cfg.train.dataset.args)
    val_ds = getattr(datasets, cfg.val.dataset.name)(**cfg.val.dataset.args)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.val.batch_size, shuffle=False, num_workers=cfg.val.num_workers, drop_last=True
    )

    # モデル
    ckpt_path = os.path.join(cfg.predictor_ckpt_dir, "model.ckpt")
    config_path = os.path.join(cfg.predictor_ckpt_dir, "config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    predictor_module = PredictorLightningModule.load_from_checkpoint(
        ckpt_path, model_name=config['model']['name'], model_args=config['model']['args'], task=config['model']['task'])
    predictor = predictor_module.model
    predictor.eval()
    pl_module = CoTGlowModule(predictor, cfg.model.args, **cfg.optimizer)

    # wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        save_dir=output_dir
    )

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=False,
        verbose=True
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch{epoch:04d}",
        every_n_epochs=10,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True
    )

    if cfg.trainer.backend is not None:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend=cfg.trainer.backend, find_unused_parameters=True)
    else:
        strategy = cfg.trainer.strategy

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=strategy,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        default_root_dir=output_dir,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
    )

    trainer.fit(pl_module, train_loader, val_loader, ckpt_path=cfg.resume)

if __name__ == "__main__":
	main()
