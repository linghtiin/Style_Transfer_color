"""
    多模型对比测试
"""
import os
import argparse
import torch as th
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image

from Source.myModels import dataset
from Source.myModels import Models
from Source.myTools.TrainTools import check_gpu, show_tensor

_hw = 512

trans = transforms.Compose([transforms.Resize((_hw, _hw)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            #                      std=(0.229, 0.224, 0.225))
                            ])


def load_model(model_type, reuse, device, noAdaIN=False, isProColor=True):
    model = model_type().to(device)
    if reuse is not None:
        model_dict = th.load(reuse, map_location=device)
        model.load_state_dict(model_dict)

    if hasattr(model, 'device'):
        model.device = device
    if hasattr(model, 'noAdaIn'):
        if noAdaIN:
            model.noAdaIn = True
        else:
            model.noAdaIn = False
    if hasattr(model, 'isProColor'):
        if isProColor:
            model.isProColor = True
        else:
            model.isProColor = False

    return model


def test(args):
    device = check_gpu(args.gpu)

    tensor_c = []
    tensor_s = []
    for i, (c, s) in enumerate(zip(args.content, args.style)):
        imgc = Image.open(c)
        imgs = Image.open(s)
        tc = trans(imgc)
        ts = trans(imgs)
        tensor_c.append(tc)
        tensor_s.append(ts)
    tensor_c = th.stack(tensor_c).to(device)
    tensor_s = th.stack(tensor_s).to(device)

    out_list = []
    for i, (model_t, reuse_t) in enumerate(zip(args.model, args.reuse)):
        model = load_model(model_t, reuse_t, device)
        model.eval()
        with th.no_grad():
            out_t = model.generate(tensor_c, tensor_s, alpha=args.alpha)

        if i == 3:
            out_t = dataset.images_camp(out_t)
        out_list.append(out_t)

    out_after = th.stack(out_list)
    _l, _r, c, h, w = out_after.shape
    # show_tensor(tensor_c)
    # show_tensor(tensor_s)
    show_tensor(out_after.reshape(-1, c, h, w))

    out_all = mk_img_mat(out_after, tensor_c, tensor_s)
    _l, _r, c, h, w = out_all.shape
    save_image(out_all.reshape(-1, c, h, w), 'mesmaek.jpg', nrow=len(args.model) + 2)


def mk_img_mat(out, tensor_c, tensor_s):
    mat_ = th.stack([tensor_c, tensor_s])
    mat_ = th.cat([mat_, out], dim=0)
    return mat_.permute(1, 0, 2, 3, 4)


class Parser:
    def __init__(self):
        self.content = None
        self.style = None
        self.alpha = 1.0
        self.gpu = 0
        self.model = [Models.Model]
        self.reuse = ['.models/pre_model_state.pth']
        self.noAdaIN = False
        self.isProColor = True


if __name__ == '__main__':
    os.chdir('./../')
    cmd_args = Parser()

    cmd_args.content = ['.dataset/My_dataset/content/blonde_girl.jpg',
                        '.dataset/My_dataset/content/brad_pitt.jpg',
                        '.dataset/My_dataset/content/neko.jpg',
                        ]

    cmd_args.style = ['.dataset/style_train/222.jpg',
                      '.dataset/My_dataset/style/hosi.jpg',
                      '.dataset/My_dataset/style/rain_princess.jpg',
                      ]

    cmd_args.model = [Models.Model,
                      Models.Model8_color1,
                      Models.Model9_color2,
                      Models.Model16_color4]
    cmd_args.reuse = ['.models/pre_model_state.pth',
                      '.models/pre_model_color1.pth',
                      '.models/pre_model_color2.pth',
                      '.models/pre_model_color4.pth', ]

    test(cmd_args)
