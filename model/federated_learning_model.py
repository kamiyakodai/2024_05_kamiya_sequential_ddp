import argparse
import torch

from utils import compute_topk_accuracy
from model import configure_model, ModelConfig
from model.simple_lightning_model import SimpleLightningModel
from typing import Optional


class FederatedLightningModel(SimpleLightningModel):
    def __init__(
            self,
            command_line_args: argparse.Namespace,
            n_classes: int,
            exp_name: str,
    ):
        super().__init__(command_line_args, n_classes, exp_name)
        self.args = command_line_args
        self.exp_name = exp_name
        # self.n_classes = n_classes
        # self.save_hyperparameters()
        self.model_list = []

    def setup(self, stage=None):
        for _ in range(self.args.model_num):
            self.model = configure_model(ModelConfig(
                model_name=self.args.model_name,
                use_pretrained=self.args.use_pretrained,
                torch_home=self.args.torch_home,
                n_classes=self.n_classes,
            ))
            self.model_list.append(self.model)
        return self.model_list

    # def on_train_batch_start(self, batch, batch_idx):
    #     # ここで複数モデルを生成してfederated learningを計算する
    #     # モデルのパラメーターを平均化し、各モデルのパラメーターを更新する
    #     for _ in range(self.args.model_num):
    #         self.model = configure_model(ModelConfig(
    #             model_name=self.args.model_name,
    #             use_pretrained=self.args.use_pretrained,
    #             torch_home=self.args.torch_home,
    #             n_classes=self.n_classes,
    #         ))
    #         self.model_list.append(self.model)
    #     return self.model_list

    def training_step(self, batch, batch_idx):

        loss_list = []
        data, labels = batch
        batch_size = data.size(0)

        for model in self.model_list:
            data = data.view([1, 0, 2, 3])
            outputs = model(data, labels=labels)
            loss = outputs.loss
            loss_list.append(self.model)

            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            self.log_train_loss_top15(loss, top1, top5, batch_size)

        return loss_list

    def on_after_backward(self):
        ratio = self.args.ratio

        with torch.no_grad():
            model_ave = {}
            for model in self.model_list:
                for k, p in model.state_dict().items():
                    if k in model_ave.keys():
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
                        p[k][:] = (p[k][:] * ratio) + (model_ave[k][:] * (1 - ratio))
        return p
