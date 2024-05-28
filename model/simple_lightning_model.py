import argparse
import os
import torch


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import SGD

from utils import compute_topk_accuracy
from model import configure_model, ModelConfig
from setup import configure_optimizer, configure_scheduler


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: argparse.Namespace,
            n_classes: int,
            exp_name: str,
            models: list,
            ratio: float,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name
        self.model_list = models
        self.ratio = ratio

        for _ in range(self.args.model_num):
            self.model = configure_model(ModelConfig(
                model_name=self.args.model_name,
                use_pretrained=self.args.use_pretrained,
                torch_home=self.args.torch_home,
                n_classes=n_classes,
            ))
            self.model_list.append(self.model)

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        scheduler_list = []
        optimizer_list = []

        # for _ in range(self.args.model_num):
        #     optimizer = SGD(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     momentum=self.args.momentum,
        #     weight_decay=self.args.weight_decay,
        # )
        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            model_params=self.model.parameters()
        )
            # optimizer_list.append(optimizer)

        scheduler = configure_scheduler(
            optimizer=optimizer,
            use_scheduler=self.args.use_scheduler
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):

        # for model in self.model_list:
        #     ratio = self.ratio

        #     videos, labels = batch

        #     outputs = model(videos, labels=labels)
        #     loss = outputs.loss
        ratio = self.ratio

        videos, labels = batch

        outputs = self.model(videos, labels=labels)
        loss = outputs.loss

        # モデルのパラメーターを平均化し、各モデルのパラメーターを更新する
        with torch.no_grad():
            model_ave = {}
            for model in self.model_list:
                for k, p in model.state_dict().items():
                    if k in model_ave.keys():
                        model_ave[k] = model_ave[k].to('cuda')
                        model_ave[k] += p.data.detach().clone()
                    else:
                        model_ave[k] = p.data.detach().clone()
            for k in model_ave.keys():
                if str(model_ave[k].dtype) != "torch.float32":
                    continue
                else:
                    model_ave[k] /= len(self.model_list)

            for model in self.model_list:
                p = model.state_dict()
                for k in model.state_dict().keys():
                    if str(model_ave[k].dtype) != "torch.float32":
                        continue
                    else:
                        p[k] = p[k].to('cuda')
                        p[k][:] = (p[k][:] * ratio) + (model_ave[k][:] * (1 - ratio))
                        # model_ave[k] = model_ave[k].to('cpu')
                        # p[k] = p[k].to('cpu')

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """a single validation step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Note:
            DO NOT USE .to() or model.eval() here
                (automatically send to multi-GPUs)
            DO NOT USE with torch.no_grad() here
                (automatically handled by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_val_loss_top15(loss, top1, top5, batch_size)
