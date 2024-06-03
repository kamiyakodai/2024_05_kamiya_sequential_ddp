import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model.simple_lightning_model import SimpleLightningModel


from dataset.dataset import dataset_factory
from args import get_args
from get_model import get_model
from get_optimizer import get_optimizer
from recognition_pretrained_AVA import recognition_accuracy, x3d_recognition_accuracy


class VideoRecognitionModel(SimpleLightningModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, self.n_classes = dataset_factory(hparams)
        self.models = get_model(hparams, self.n_classes, self.device, self.models)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.global_step_train = 0
        self.global_step_val = 0

    def training_step(self, batch, batch_idx):
        data, labels = self.process_batch(batch)

        if self.hparams.pretrained_model == "MAE":
            data = data.permute(0, 2, 1, 3, 4)
            output = self(data)
            logits = output.logits
            train_loss_ce = self.criterion_ce(logits, labels)
        else:
            output = self(data)
            train_loss_ce = self.criterion_ce(output, labels)

        self.log('train_loss', train_loss_ce, prog_bar=True)
        return train_loss_ce

    def process_batch(self, batch):
        if self.hparams.dataloader == "pytorchvideo":
            videos = batch["video"].to(self.device)
            labels = batch["label"].to(self.device)
        elif self.hparams.dataloader == "webdataset":
            videos = batch[0].to(self.device)
            labels = batch[1]["label"].to(self.device)
            videos = videos.transpose(1, 2).permute(0, 2, 1, 3, 4)
        elif self.hparams.dataloader == "sequential":
            videos = torch.stack(batch[0], dim=0).transpose(1, 2).to(self.device)
            labels = torch.tensor(batch[1]).to(self.device)
        else:
            raise ValueError(f"Unknown dataloader: {self.hparams.dataloader}")

        return videos, labels

    def validation_step(self, batch, batch_idx):
        data, labels = self.process_batch(batch)

        if self.hparams.pretrained_model == "MAE":
            data = data.permute(0, 2, 1, 3, 4)
            output = self(data)
            logits = output.logits
            loss = output.loss
        else:
            output = self(data)
            loss = output.loss

        if self.hparams.pretrained_model == "MAE":
            # val_top1, val_top5 = recognition_accuracy(output, labels, topk=(1, 5))
            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        else:
            # val_top1, val_top5 = x3d_recognition_accuracy(output, labels, topk=(1, 5))
            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))

        return loss
