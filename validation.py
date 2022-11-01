import torch
from config.train_config import TrainConfig
from data.dataprocess import Dataset
from torch.utils import data
from models.PConv import PCConv
from models.network.pconv import Encoder,Decoder
from models import PConv
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import numpy as np
from loguru import logger
from models.models import create_model

def get_grid_images(visdict, size=256, dim=1,):
    """
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    """
    assert dim == 1 or dim == 2
    grids = {}
    for key in visdict:
        _, _, h, w = visdict[key].shape
        if dim == 2:
            new_h = size
            new_w = int(w * size / h)
        elif dim == 1:
            new_h = int(h * size / w)
            new_w = size
        grids[key] = torchvision.utils.make_grid(
            F.interpolate(visdict[key], [new_h, new_w]).detach().cpu()
        )
    grid = (torch.cat(list(grids.values()), 1) + 1) / 2
    grid_image = (grid.numpy().transpose(
        1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    return grid_image


plt.figure(figsize=(8, 8))

cfg = TrainConfig().create_config()
pconvModel = create_model(cfg)
# for name,param in pconvModel.named_parameters():
#     print(name)
#
dataset = Dataset(cfg.gt_root, cfg.mask_root, cfg)
iterator_val = data.DataLoader(
    dataset, batch_size=cfg.batchSize, shuffle=True, num_workers=cfg.num_workers
)
# TODO: validate pconv model