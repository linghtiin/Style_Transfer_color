import functools
import torch as th
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from Source.myModels.dataset import denorm, PreprocessDataset


def mk_train_dir(save_path):
    dir_save = Path(save_path)
    dir_loss = dir_save / 'loss'
    dir_image = dir_save / 'image'
    dir_model_state = dir_save / 'model_state'

    for path in [dir_image, dir_loss, dir_model_state]:
        if not path.exists():
            path.mkdir(parents=True)

    return dir_image, dir_loss, dir_model_state


def mk_save_dir(save_path):
    dir_save = Path(save_path)
    if not dir_save.exists():
        dir_save.mkdir(parents=True)
    return dir_save


def check_gpu(gpu):
    # set device on GPU if available, else CPU
    if th.cuda.is_available() and gpu >= 0:
        device = th.device(f'cuda:{gpu}')
        print(f'# CUDA available: {th.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    return device


def load_model(args, device):
    print('To start load model.')
    model = args.model().to(device)
    if args.reuse is not None:
        model_dict = th.load(args.reuse, map_location=device)
        for k, v in model.named_parameters():
            if k in model_dict.keys():
                v.data = model_dict[k].data
            else:
                print(f'{k} is missing a Train data, it is re-initially.')
                if k.find("bias") >= 0:
                    th.nn.init.constant_(v.data, 0)  # bias 初始化为0
                else:
                    th.nn.init.xavier_normal_(v.data)  # 没有预训练，则使用xavier初始化

    if hasattr(model, 'device'):
        model.device = device
    if hasattr(model, 'noAdaIn'):
        if args.noAdaIN:
            model.noAdaIn = True
        else:
            model.noAdaIn = False
    if hasattr(model, 'isProColor'):
        if args.isProColor:
            model.isProColor = True
        else:
            model.isProColor = False

    return model


def mk_loss_plt(save_path, losslist):
    plt.plot(range(len(losslist)), losslist)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{save_path}/train_loss.png')
    with open(f'{save_path}/loss_log.txt', 'w') as f:
        for _loss in losslist:
            f.write(f'{_loss}\n')
    print(f'Loss saved in {save_path}')

    return sum(losslist) / len(losslist)


def log_test(content_dir, style_dir, model, device):
    test_dataset = PreprocessDataset(content_dir, style_dir)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    for i, (c, s) in enumerate(test_loader):
        content = c.to(device)
        style = s.to(device)
        with th.no_grad():
            out = model.generate(content, style)
        content = th.cat([content[i] for i in range(content.shape[0])], dim=1)
        style = th.cat([style[i] for i in range(style.shape[0])], dim=1)
        out = th.cat([out[i] for i in range(out.shape[0])], dim=1)
        res = th.cat([content, style, out], dim=2)
        yield res


def tensor_to_image(tensor):
    t = tensor.cpu() * 255
    if t.ndim == 3:
        arr = np.uint8(t).transpose(1, 2, 0)
        return Image.fromarray(arr)
    elif t.ndim == 4:
        arr = np.uint8(t).transpose(0, 2, 3, 1)
        imgs = []
        for img in arr:
            imgs.append(Image.fromarray(img))
        return imgs
    else:
        raise


def show_tensor(tensor):
    img = tensor_to_image(tensor)
    if type(img) is list:
        for i in img:
            plt.imshow(i)
            plt.show()
    else:
        plt.imshow(img)
        plt.show()
