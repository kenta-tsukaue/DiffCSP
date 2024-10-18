import os
from pathlib import Path
from typing import List
import sys
sys.path.append('.')
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from diffcsp.pl_data.datamodule import CrystDataModule
from diffcsp.pl_modules.cspnet import CSPNet
print(CrystDataModule)

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT

import wandb

os.environ['HYDRA_FULL_ERROR'] = '1'




def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """++++
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
        # 確認したいディレクトリのパス
    directory_path = hydra_dir / 'lattice_scaler.pt'

    # ディレクトリが存在しない場合は作成する
    if not os.path.exists(directory_path.parent):
        os.makedirs(directory_path.parent)
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')


    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    #これはeasy_structure_2
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-08-10/train_d1_43/epoch=33724-step=1349000.ckpt"
    #これはeasy_structure_3
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-09-03/train_d1_44/epoch=29634-step=1185400.ckpt"
    #これはeasy_structure_4
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-09-13/train_d1_46/epoch=18539-step=741600.ckpt"

    # 48: perov_5
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-09-18/train_d1_48/epoch=924-step=11100.ckpt"

    # 49: carbon_24
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-09-18/train_d1_49/epoch=5359-step=128640.ckpt"

    # 50: mpts_52
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-09-18/train_d1_50/epoch=469-step=100580.ckpt"

    # 51: Cu3Au
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-10-16/train_d1_51/epoch=11729-step=469200.ckpt"

    # 51_2: Cu3Au
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-10-16/train_d1_51_2/epoch=3264-step=130600.ckpt"

    # 51_3: Cu3Au
    #ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-10-16/train_d1_51_3/epoch=8909-step=356400.ckpt"

    # 51_4: Cu3Au
    ckpt = "/public/tsukaue/DiffCSP/hydra/singlerun/2024-10-16/train_d1_51_4/epoch=14809-step=592400.ckpt"

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.to("cuda")

    datamodule.setup()
    #test_dataloader = datamodule.test_dataloader()[0] # test data
    #test_dataloader = datamodule.val_dataloader()[0] # val data
    test_dataloader = datamodule.train_dataloader(shuffle = False) # train data
    for batch_idx, batch in enumerate(test_dataloader):
        batch = batch.to("cuda")
        print(f"Test Batch {batch_idx + 1}: {batch}")
        """#traj, traj_stack, batch_loss_list, loss_list = model.sample_new_method(batch)
        traj, traj_stack, batch_loss_list, loss_list = model.sample(batch)
        # Save traj and batch to files
        loss_tensor = torch.tensor(loss_list)
        batch_loss_tensor = torch.tensor(batch_loss_list)
        torch.save(loss_tensor, 'loss_tensor.pt')
        torch.save(batch_loss_tensor, 'batch_loss_tensor.pt')
        torch.save(traj, 'traj.pt')
        torch.save(batch, 'batch.pt')"""

        traj, traj_new, batch_loss_list, loss_list, batch_loss_list_new, loss_list_new = model.sample_new_method_and_random(batch)
        loss_tensor = torch.tensor(loss_list)
        batch_loss_tensor = torch.tensor(batch_loss_list)
        loss_tensor_new = torch.tensor(loss_list_new)
        batch_loss_tensor_new = torch.tensor(batch_loss_list_new)
        torch.save(loss_tensor, f'loss_tensor_{batch_idx}.pt')
        torch.save(loss_tensor_new, f'loss_tensor_new_{batch_idx}.pt')
        torch.save(batch_loss_tensor, f'batch_loss_tensor_{batch_idx}.pt')
        torch.save(batch_loss_tensor_new, f'batch_loss_tensor_new_{batch_idx}.pt')
        torch.save(traj, f'traj_{batch_idx}.pt')
        torch.save(traj_new, f'traj_new_{batch_idx}.pt')
        torch.save(batch, f'batch_{batch_idx}.pt')

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1" )
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()