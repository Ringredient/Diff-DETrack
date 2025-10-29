import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    print("--- instantiating ---")
    dm = instantiate(cfg.datamodule)
    model = instantiate(cfg.multitask)
    logger = WandbLogger(
        project="particle_tracking",
        name=f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_dir="../logs/",
        log_model=True,
    )
    callbacks = [instantiate(cb) for cb in cfg.callbacks]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    print("--- start training ---")
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt_path)
    # trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
