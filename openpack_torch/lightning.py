from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.functional import accuracy as accuracy_score
from torchmetrics.functional import f1_score
logger = getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    log_dict: Dict[str, List] = {"train": [], "val": [], "test": []}
    log_keys: Tuple[str, ...] = ("loss", "acc")

    def __init__(self, cfg: DictConfig = None) -> None:
        self.cfg = cfg
        super().__init__()

        self.net: nn.Module = self.init_model(cfg)
        self.criterion: nn.Module = self.init_criterion(cfg)

    def init_model(self, cfg) -> torch.nn.Module:
        raise NotImplementedError()

    def init_criterion(self, cfg: DictConfig):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.cfg.train.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.train.optimizer.multistep_milestones,
                gamma=self.cfg.train.optimizer.multistep_gamma,
            )
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        elif self.cfg.train.optimizer.type == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.train.optimizer.lr,
                momentum=self.cfg.train.optimizer.momentum,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
            
        else:
            raise ValueError(
                f"{self.cfg.train.optimizer.type} is not supported.")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def calc_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns accuracy score.

        Args:
            y (torch.Tensor): logit tensor. shape=(BATCH, CLASS, TIME), dtype=torch.float
            t (torch.Tensor): target tensor. shape=(BATCH, TIME), dtype=torch.long

        Returns:
            torch.Tensor: _description_
        """
        preds = F.softmax(y, dim=1)
        (batch_size, num_classes, window_size) = preds.size()
        preds_flat = preds.permute(1, 0, 2).reshape(
            num_classes, batch_size * window_size)
        t_flat = t.reshape(-1)

        # FIXME: I want to use macro average score.
        ignore_index = num_classes - 1
        acc = accuracy_score(
            preds_flat.transpose(0, 1),
            t_flat,
            task="multiclass",
            average="weighted",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        log = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                log[f"train/{key}"] = torch.stack(vals).mean().item()
        self.log_dict["train"].append(log)

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        (
            x_imu,
            t_imu,
            ts_imu,
            x_e4acc,
            x_keypoints,
            x_bbox,
            x_ht,
            x_printer
        ) = self.split_data(batch)

        if self.v2:
            y_hat = self.net(x_imu,
                x_e4acc,
                x_keypoints,
                x_bbox,
                x_ht,
                x_printer).squeeze(3)
        else:
            y_hat = self.net(x_imu,
                x_e4acc,
                x_keypoints,
                x_bbox,
                x_ht,
                x_printer).squeeze(3)
        
        loss = self.criterion(y_hat, t_imu)
        acc = self.calc_accuracy(y_hat, t_imu)

        if batch_idx == 0:
            self.val_y_batch = []
            self.val_t_batch = []
        self.val_y_batch.append(y_hat)
        self.val_t_batch.append(t_imu)
        return {"loss": loss, "acc": acc}
    
        #return self.training_step(batch, batch_idx)
    

    def adjust_learning_rate(self):
        #Se ajustaran el learning rate dependiendo la epoca contenida en params
        opt = self.optimizers()
        current_epoch = self.current_epoch
        for epoch, lr, wd in self.cfg.learningData:
            if(epoch == current_epoch):
                opt.param_groups[0]['lr'] = lr;
                opt.param_groups[0]['weight_decay'] = wd;
        
    def validation_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            # When multiple dataloader is used.
            _outputs = []
            for out in outputs:
                _outputs += out
            outputs = _outputs

        log = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                avg = torch.stack(vals).mean().item()
                log[f"val/{key}"] = avg
        
        if(self.cfg.dynamicLr):
            self.adjust_learning_rate()
        
        val_y = torch.concat(self.val_y_batch)
        val_t = torch.concat(self.val_t_batch)
        f1macro = self.calc_f1macro(val_y, val_t)
        log["val/f1macro"] = f1macro

        self.log_dict["val"].append(log)

        self.print_latest_metrics()

        if len(self.log_dict["val"]) > 0:
            val_loss = self.log_dict["val"][-1].get("val/loss", None)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            val_acc = self.log_dict["val"][-1].get("val/acc", None)
            self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_f1macro", f1macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def calc_f1macro(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        preds = F.softmax(y, dim=1)
        (batch_size, num_classes, window_size) = preds.size()
        preds_flat = preds.permute(1, 0, 2).reshape(num_classes, batch_size * window_size)
        t_flat = t.reshape(-1)

        ignore_index = num_classes - 1
        f1macro = f1_score(
            preds_flat.transpose(0, 1),
            t_flat,
            task="multiclass",
            average="macro",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        return f1macro
    
    
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        keys = tuple(outputs[0].keys())
       
        results = {key: [] for key in keys}
        for d in outputs:
            for key in d.keys():
                results[key].append(d[key].cpu().numpy())

        for key in keys:
            results[key] = np.concatenate(results[key], axis=0)

        self.test_results = results

    def print_latest_metrics(self) -> None:
        # -- Logging --
        train_log = self.log_dict["train"][-1] if len(
            self.log_dict["train"]) > 0 else dict()
        val_log = self.log_dict["val"][-1] if len(
            self.log_dict["val"]) > 0 else dict()
        log_template = (
            "Epoch[{epoch:0=3}]"
            " TRAIN: loss={train_loss:>7.4f}, acc={train_acc:>7.4f}"
            " | VAL: loss={val_loss:>7.4f}, acc={val_acc:>7.4f}, f1macro={val_f1macro:>7.4f}"
            " | LR: {lr:>7.4f}"
            " | WD: {wd:>7.4f}"
        )
        opt = self.optimizers()
        logger.info(
            log_template.format(
                epoch=self.current_epoch,
                train_loss=train_log.get("train/loss", -1),
                train_acc=train_log.get("train/acc", -1),
                val_loss=val_log.get("val/loss", -1),
                val_acc=val_log.get("val/acc", -1),
                val_f1macro=val_log.get("val/f1macro", -1),
                lr=opt.param_groups[0]['lr'],
                wd=opt.param_groups[0]['weight_decay'],
            )
        )



class BaseImageLightningModule(pl.LightningModule):
    log_dict: Dict[str, List] = {"train": [], "val": [], "test": []}
    log_keys: Tuple[str, ...] = ("loss", "acc")

    def __init__(self, cfg: DictConfig = None) -> None:
        self.cfg = cfg
        super().__init__()

        self.net: nn.Module = self.init_model(cfg)
        self.criterion: nn.Module = self.init_criterion(cfg)

    def init_model(self, cfg) -> torch.nn.Module:
        raise NotImplementedError()

    def init_criterion(self, cfg: DictConfig):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.cfg.train.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.cfg.train.optimizer.type} is not supported.")
        return optimizer

    def calc_accuracy(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns accuracy score.

        Args:
            y (torch.Tensor): logit tensor. shape=(BATCH, CLASS, TIME), dtype=torch.float
            t (torch.Tensor): target tensor. shape=(BATCH, TIME), dtype=torch.long

        Returns:
            torch.Tensor: _description_
        """
        preds = F.softmax(y, dim=1)
    
        (batch_size, num_classes) = preds.size()
        preds_flat = preds.permute(1, 0).reshape(
            num_classes, batch_size)
        t_flat = t.reshape(-1)

        # FIXME: I want to use macro average score.
        ignore_index = num_classes - 1
        acc = accuracy_score(
            preds_flat.transpose(0, 1),
            t_flat,
            task="multiclass",
            average="weighted",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        log = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                log[f"train/{key}"] = torch.stack(vals).mean().item()
        self.log_dict["train"].append(log)

    def validation_step(
            self,
            batch: Dict,
            batch_idx: int,
            dataloader_idx: int = 0) -> Dict:
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            # When multiple dataloader is used.
            _outputs = []
            for out in outputs:
                _outputs += out
            outputs = _outputs

        log = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                avg = torch.stack(vals).mean().item()
                log[f"val/{key}"] = avg
        self.log_dict["val"].append(log)

        self.print_latest_metrics()

        if len(self.log_dict["val"]) > 0:
            val_loss = self.log_dict["val"][-1].get("val/loss", None)
            self.log(
                "val/loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True)

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        keys = tuple(outputs[0].keys())
       
        results = {key: [] for key in keys}
        for d in outputs:
            for key in d.keys():
                results[key].append(d[key].cpu().numpy())

        for key in keys:
            results[key] = np.concatenate(results[key], axis=0)

        self.test_results = results

    def print_latest_metrics(self) -> None:
        # -- Logging --
        train_log = self.log_dict["train"][-1] if len(
            self.log_dict["train"]) > 0 else dict()
        val_log = self.log_dict["val"][-1] if len(
            self.log_dict["val"]) > 0 else dict()
        log_template = (
            "Epoch[{epoch:0=3}]"
            " TRAIN: loss={train_loss:>7.4f}, acc={train_acc:>7.4f}"
            " | VAL: loss={val_loss:>7.4f}, acc={val_acc:>7.4f}"
        )
        logger.info(
            log_template.format(
                epoch=self.current_epoch,
                train_loss=train_log.get("train/loss", -1),
                train_acc=train_log.get("train/acc", -1),
                val_loss=val_log.get("val/loss", -1),
                val_acc=val_log.get("val/acc", -1),
            )
        )