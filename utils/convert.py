import numpy as np
import torch


def change_to_classify(y):
    # 确保y是torch.Tensor类型
    if not isinstance(y, torch.Tensor):
        raise ValueError("y must be a torch.Tensor")

    # 使用torch.where进行条件选择，返回对应的分类标签
    classification_label = torch.where(y < 0, torch.zeros_like(y),
                                 torch.where(y == 0, torch.ones_like(y),
                                             torch.ones_like(y) * 2)).long()

    return classification_label