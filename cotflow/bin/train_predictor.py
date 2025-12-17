import os
import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import cotflow.datasets as datasets
import cotflow.nets.predictor as predictor
import cotflow.nets.autoencoder as autoencoder
from cotflow.nets.autoencoder import AutoencoderModule
from cotflow.nets.predictor import PredictorLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import cotflow.utils as utils


@hydra.main(config_path=os.path.abspath("config"), config_name='predictor_attr.yaml', version_base=None)
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
    if cfg.autoencoder_ckpt_dir is not None:
        # ckpt と config　のパスを特定
        ckpt_path = os.path.join(cfg.autoencoder_ckpt_dir, "model.ckpt")
        config_path = os.path.join(cfg.autoencoder_ckpt_dir, "config.yaml")

        ckpt = torch.load(ckpt_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        autoencoder_model = getattr(autoencoder, config['model']['name'])(**config['model']['args'])
        autoencoder_module = AutoencoderModule(autoencoder_model)
        autoencoder_module.load_state_dict(ckpt['state_dict'])
        autoencoder_model = autoencoder_module.model
        autoencoder_model.eval()
    else:
        autoencoder_model = None

    model = getattr(predictor, cfg.model.name)(**cfg.model.args)
    pl_module = PredictorLightningModule(model, autoencoder=autoencoder_model, task=cfg.model.task, lr=cfg.optimizer.lr)

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

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() and cfg.trainer.gpus > 0 else 'cpu',
        devices=cfg.trainer.gpus if torch.cuda.is_available() and cfg.trainer.gpus > 0 else 1,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        default_root_dir=output_dir,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(pl_module, train_loader, val_loader, ckpt_path=cfg.resume)

if __name__ == "__main__":
    main()
