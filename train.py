from dataclasses import dataclass

import comet_ml
import torch
import torch.nn as nn
import numpy as np
import random

from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import (
    compute_topk_accuracy,
    AvgMeterLossTopk,
    TqdmLossTopK,
)
from model import ClassificationBaseModel, FederatedLearningModel, get_device


@dataclass
class TrainConfig:
    grad_accum_interval: int
    log_interval_steps: int


@dataclass
class TrainOutput:
    loss: float
    top1: float
    train_step: int


class LoggerChecker:
    def __init__(self, log_interval_steps):
        self.log_interval_steps = log_interval_steps

    def should_log(self, step):
        return step % self.log_interval_steps == 0


class OptimizerChecker:
    def __init__(self, grad_accum_interval):
        self.grad_accum_interval = grad_accum_interval

    def should_zero_grad(self, batch_index):
        return (
            self.grad_accum_interval == 1
            or batch_index % self.grad_accum_interval == 1
        )

    def should_update(self, batch_index):
        return batch_index % self.grad_accum_interval == 0

# モデル，オプティマイザーをリスト化する必要あり


def train(
    model: FederatedLearningModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    current_train_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
    train_config: TrainConfig,
    epoch,
    experiment,
    global_step,
    args,
) -> TrainOutput:
    """training loop for one epoch

    Args:
        model (ClassificationBaseModel): CNN model
        optimizer (Optimizer): optimizer
        scheduler (LRScheduler): learning rate (lr) scheduler
        loader (DataLoader): training dataset loader
        current_train_step (int): current step for training
        current_epoch (int): current epoch
        logger (comet_ml.Experiment): comet logger
        train_config (TrainInfo): information for training

    Returns:
        TrainOutput: train loss, train top1, steps for training
    """

    train_losses = []
    train_top1s = []
    train_top5s = []
    train_log_federate_loss_ce = []

    models = []
    for i in range(args.model_num):
        models[i].to(device)
        models[i].train()

    train_meters = AvgMeterLossTopk("train")

    model.train()
    device = get_device(model)
    optimizer_checker = OptimizerChecker(train_config.grad_accum_interval)
    logger_checker = LoggerChecker(train_config.log_interval_steps)

    with TqdmLossTopK(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            leave=False,
            unit='step',
    ) as progress_bar_step:
        progress_bar_step.set_description("[train    ]")

        for batch_index, batch in progress_bar_step:
            model_idx = batch_index % args.model_num
            batch_size = args.batch_size

            # data, labels = batch  # (BCHW, B) for images or (BCTHW, B) for videos
            # data = data.to(device)
            # labels = labels.to(device)
            data, labels = make_data_labels(args, device, batch)

            batch_size = data.size(0)

            if optimizer_checker.should_zero_grad(batch_index):
                optimizer.zero_grad()

            # outputs = model(data, labels=labels)
            # loss = outputs.loss.mean()  # maen() is only for dp to gather loss
            # loss.backward()
            outputs, loss = loss_calc(args, model, data, labels, optimizer, batch_size,epoch ,global_step)

            train_topk = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            train_meters.update(loss, train_topk, batch_size)

            if logger_checker.should_log(current_train_step):
                progress_bar_step.set_postfix_str_loss_topk(
                    current_train_step, loss, train_topk
                )
                logger.log_metrics(
                    train_meters.get_step_metrics_dict(),
                    step=current_train_step,
                    epoch=current_epoch,
                )

            if optimizer_checker.should_update(batch_index):
                optimizer.step()
                current_train_step += 1

    scheduler.step()

    logger.log_metrics(
        train_meters.get_epoch_metrics_dict(),
        step=current_train_step,
        epoch=current_epoch,
    )

    return TrainOutput(
        loss=train_meters.loss_meter.avg,
        top1=train_meters.topk_meters[0].avg,  # top1: topk[0] should be 1
        train_step=current_train_step
    )


def make_data_labels(args, device, batch):
    if args.dataloader == "pytorchvideo":
        data = batch["video"].to(device)
        labels = batch["label"].to(device)
    elif args.dataloader == "webdataset":
        if args.dataset == "UCF":
            data = batch[0].to(device)
            labels = batch[1]["label"].to(device)
        if args.dataset == "HMDB":
            data = batch[0].to(device)
            labels = batch[1]["label"].to(device)
        elif args.dataset == "MPII":
            data = batch[0].to(device)
            labels = batch[1].to(device)
    elif args.dataloader == "sequential":
        if args.dataset == "50Salads":
            for i in range(len(batch[0])):
                if i == 0:
                    data = batch[0][i].to(device)
                elif i == 1:
                    a = batch[0][i].to(device)
                    data = torch.stack(
                        (data, a), dim=0
                    )
            labels = np.ndarray(batch[1]).to(device)
        else:
            data_list = []
            for i in range(len(batch[0])):
                data_list.append(batch[0][i])
            data = torch.stack(data_list, dim=0).to(device)
            labels = torch.Tensor(batch[1])
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
    return data, labels


def loss_calc(args, models, data, labels, optimizers, batch_size, experiment, epoch, global_step):
    criterion_ce = nn.CrossEntropyLoss()
    outputs = []
    train_loss_ce_list = []
    train_log_loss_ce = []
    ratio = args.ratio

    if args.loss_calculation == "nomal":
        optimizers[0].zero_grad()
        output = models[0](data)
        train_loss_ce = criterion_ce(output, labels)
        train_loss_ce_list.append(train_loss_ce)
        train_loss_ce.backward()

        train_log_loss_ce[0].update(train_loss_ce_list[0], batch_size)
        experiment.log_metric(
            "train_batch_loss",
            train_log_loss_ce[0].val,
            epoch=epoch,
            step=global_step,
        )
        optimizers[0].step()

    elif args.loss_calculation == "federated":
        shuffle_index = list(range(batch_size))
        batch_sample_index = []
        for i in range(args.model_num):
            batch_sample_index.append(shuffle_index[i:: args.model_num])
            random.shuffle(batch_sample_index)

        for i, model in enumerate(models):
            optimizers[i].zero_grad()
            if args.model_name == "X3D":
                output = model(data[batch_sample_index[i]])
            elif args.model_name == "timesformer":
                video_input = data[batch_sample_index[i]].permute(
                    0, 2, 1, 3, 4
                )
                output = model(video_input)
                output = output.logits
            elif args.model_name == "videoMAE":
                video_input = data[batch_sample_index[i]].permute(
                    0, 2, 1, 3, 4
                )  # BCFTH -> BFCTH
                output = model(pixel_values=video_input)
                output = torch.mean(input=output.last_hidden_state, dim=1)

            train_federate_loss_ce = criterion_ce(
                output, labels[batch_sample_index[i]]
            )

            train_federate_loss_ce.backward()

            train_log_loss_ce[i].update(train_federate_loss_ce, batch_size)
            experiment.log_metric(
                str(i) + "train_batch_loss",
                train_log_loss_ce[i].val,
                epoch=epoch,
                step=global_step,
            )
            outputs.append(output)

            optimizers[i].step()

        with torch.no_grad():
            model_ave = {}
            for model in models:
                for k, p in model.state_dict().items():
                    if k in model_ave.keys():
                        model_ave[k] += p.data.detach().clone()
                    else:
                        model_ave[k] = p.data.detach().clone()
            for k in model_ave.keys():
                if str(model_ave[k].dtype) != "torch.float32":
                    continue
                else:
                    model_ave[k] /= args.model_num

            for model in models:
                p = model.state_dict()
                for k in model.state_dict().keys():
                    if str(model_ave[k].dtype) != "torch.float32":
                        continue
                    else:
                        p[k][:] = (p[k][:] * ratio) + (
                            model_ave[k][:] * (1 - ratio)
                        )
