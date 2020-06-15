"""
    多图像-单模型对比测试
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
from Source.myTools.TrainTools import check_gpu, load_model, show_tensor  # , show_histc

_hw = 512

trans = transforms.Compose([transforms.Resize((_hw, _hw)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            #                      std=(0.229, 0.224, 0.225))
                            ])


def img_afterpost(tensor):
    t = tensor
    # t = dataset.images_camp(t)
    return t


def img_trans(images_path):
    _Images = []
    for img_p in images_path:
        img = Image.open(img_p)
        tensor = trans(img)
        _Images.append(tensor)
    return th.stack(_Images)


def mk_tensor_grad(tensor_c, tensor_s):
    """
    table the images, to be (c1,c2,c3,c1,c2,c3,c1,c2,c3) and (s1,s1,s1,s2,s2,s2,s3,s3,s3)
    :param tensor_c: the images for content with (b1, c, h, w)
    :param tensor_s: the images for style with (b2, c, h, w)
    :return: the two tensor with (b1xb2, c, h, w)
    """
    b1, _, _, _ = tensor_c.shape
    b2, c, h, w = tensor_s.shape
    tc = tensor_c.repeat(b2, 1, 1, 1)
    ts_1 = tensor_s.unsqueeze(1)
    ts_2 = ts_1.repeat(1, b1, 1, 1, 1)
    ts = ts_2.view(-1, c, h, w)

    return tc, ts


def mk_img_mat(out, tensor_c, tensor_s):
    """   table the output images, to be   (emp,   c1,      ...,  cN;
                                             s1,   o1,      ...,  oN;
                                             s2, oN+1,      ..., o2N;
                                                    ......
                                             sM, o(M-1)N+1, ..., oMN;)

    :param tensor_c: the tensor for model original content in, with NxCxHxW
    :param tensor_s: the tensor for model original style in, with MxCxHxW
    :param out: the tensor for model out, with MNxCxHxW
    :return: A tensor for mat images with BsxCxHxW
    """
    empty = th.zeros_like(tensor_c[0])
    all_img = [empty]
    len_imgc = tensor_c.shape[0]
    len_imgs = tensor_s.shape[0]
    for i in range(len_imgc):
        all_img.append(tensor_c[i])
    for i in range(len_imgs):
        all_img.append(tensor_s[i])
        for ii in range(len_imgc):
            all_img.append(out[len_imgc * i + ii])
    all_img = th.stack(all_img)
    return all_img


def test(args):
    device = check_gpu(args.gpu)

    model = load_model(args, device)
    model.eval()

    tensor_c = img_trans(args.content).to(device)
    tensor_s = img_trans(args.style).to(device)

    tg_c, tg_s = mk_tensor_grad(tensor_c, tensor_s)

    with th.no_grad():
        out = model.generate(tg_c, tg_s, alpha=args.alpha)

    out_after = args.afterpost(out)
    # show_tensor(tensor_c)
    # show_tensor(tensor_s)
    # show_tensor(out_after)

    out_all = mk_img_mat(out_after, tensor_c, tensor_s)
    save_image(out_all, 'meso.jpg', nrow=len(args.content) + 1)


class Parser:
    def __init__(self):
        self.content = None
        self.style = None
        self.afterpost = img_afterpost
        self.alpha = 1.0
        self.gpu = 0
        self.model = Models.Model
        self.reuse = '.models/pre_model_state.pth'
        self.noAdaIN = False
        self.isProColor = True


if __name__ == '__main__':
    os.chdir('./../')
    cmd_args = Parser()
    # cmd_args.noAdaIN = True
    # cmd_args.isProColor = False
    # cmd_args.alpha = 0
    # cmd_args.gpu = -1

    cmd_args.content = ['.dataset/My_dataset/content/golden_gate.jpg',
                        '.dataset/My_dataset/content/brad_pitt.jpg',
                        '.dataset/My_dataset/content/neko.jpg',
                        ]

    cmd_args.style = ['.dataset/style_train/209.jpg',
                      '.dataset/My_dataset/style/hosi.jpg',
                      '.dataset/My_dataset/style/rain_princess.jpg',
                      ]

    # cmd_args.style = ['.dataset/My_dataset/style/contrast_of_forms.jpg',
    #                   '.dataset/My_dataset/style/contrast_of_forms_gray.jpg',
    #                   ]

    # cmd_args.style = ['.dataset/My_dataset/style/_resized/contrast_of_forms.bmp',
    #                   '.dataset/My_dataset/style/_resized/la_muse.bmp',
    #                   '.dataset/My_dataset/style/_resized/picasso_self_portrait.bmp',
    #                   ]

    cmd_args.model = Models.Model16_color4
    cmd_args.reuse = '.models/pre_model_color4.pth'

    test(cmd_args)
