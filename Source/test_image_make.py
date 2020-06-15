import os
import visdom
import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms as th_trans

from Source.myModels.Models import Model
from Source.myModels.dataset import color_transfer, color_transfer_reinhard, norm, denorm
from Source.myTools.TrainTools import show_tensor, check_gpu

trans = th_trans.Compose([
    th_trans.RandomCrop(512),
    th_trans.ToTensor(),
    # th_trans.Normalize(mean=my_format[0],
    #                    std=my_format[1])
])


def etc_01(c1, c2, s1, s2):
    tc = th.cat([c1, c1, c2, c2], dim=0)
    ts = th.cat([s1, s2, s1, s2], dim=0)
    tensor_out = color_transfer(tc, ts)
    return tensor_out


def etc_02(c1, c2, s1, s2):
    tc = th.cat([c1, c1, c2, c2], dim=0)
    ts = th.cat([s1, s2, s1, s2], dim=0)
    tensor_out = color_transfer_reinhard(tc, ts)
    return tensor_out


def etc_03(c1, c2, s1, s2):
    tc = th.cat([c1, c1, c2, c2], dim=0)
    ts = th.cat([s1, s2, s1, s2], dim=0)
    ts1 = color_transfer_reinhard(tc, ts)
    tensor_out = color_transfer(tc, ts1)
    return tensor_out


def etc_04(c1, c2, s1, s2):
    tc = th.cat([c1, c1, c2, c2], dim=0)
    ts = th.cat([s1, s2, s1, s2], dim=0)

    tensor_out = model_test(Model, '.models/pre_model_state.pth', tc, ts, 1.0)
    return tensor_out


def show_sample(out, c1, c2, s1, s2, title='Image Color histogram matching'):
    tc = th.cat([c1, c1, c2, c2], dim=0)
    ts = th.cat([s1, s2, s1, s2], dim=0)
    mk_tensor = [th.ones_like(tc[0]), ts[0], ts[1],
                 tc[0], out[0], out[1],
                 tc[2], out[2], out[3]]
    out_list = th.stack(mk_tensor)
    vis.images(out_list, nrow=3, opts=dict(title=title, ))


def show_histc(tensor):
    _title = 'Histogram for Tensor'
    _dim = tensor.ndim
    h, w = tensor.shape[-2:]
    _bins = min(256, h * w)
    _minx, _maxx = tensor.min(), tensor.max()
    linrange = np.linspace(_minx, _maxx, _bins)

    if _dim == 3:
        c, _ = tensor.shape[:2]
        _tensor = tensor.view(c, -1)
        _X = []
        for i in range(c):
            _X.append(tensor[i].histc(bins=_bins, min=_minx, max=_maxx))
        vis.bar(th.stack(_X, dim=-1), linrange, opts=dict(title=_title, legend=['R', 'G', 'B']))
    elif _dim == 4:
        b, c = tensor.shape[:2]
        _tensor = tensor.view(b, c, -1)
        for ib in range(b):
            _Channel = []
            for ic in range(c):
                _Channel.append(tensor[ib, ic].histc(bins=_bins, min=_minx, max=_maxx))
            _Channel = th.stack(_Channel, dim=-1)
            vis.bar(_Channel, linrange, opts=dict(title=_title + ' in Sample_%d' % ib, legend=['R', 'G', 'B']))
    elif _dim == 5:
        b, m, c = tensor.shape[:3]
        _tensor = tensor.view(b, m, c, -1)
        for ib in range(b):
            _Batch = []
            _legend = []
            for im in range(m):
                for ic in range(c):
                    _Batch.append(tensor[ib, im, ic].histc(bins=_bins, min=_minx, max=_maxx))
                _legend += ['Match%02d_R' % im, 'Match%02d_G' % im, 'Match%02d_B' % im]
            _Batch = th.stack(_Batch, dim=-1)
            vis.bar(_Batch, linrange, opts=dict(title=_title + ' in Sample_%d' % ib, legend=_legend))


def model_test(model_type, reuse, content, style, alpha):
    device = check_gpu(0)
    for i, (c, s) in enumerate(zip(content, style)):
        content[i] = norm(c)
        style[i] = norm(s)
    model = model_type().to(device)
    model.load_state_dict(th.load(reuse, map_location=device))
    model.eval()

    content = content.to(device)
    style = style.to(device)
    with th.no_grad():
        out = model.generate(content, style, alpha=alpha)
    out = denorm(out, device)
    out = out.cpu()
    return out


if __name__ == '__main__':
    vis = visdom.Visdom(env='image_transfer')
    os.chdir('./../')

    img1 = Image.open('.dataset/My_dataset/content_t1/test-1.bmp')
    img2 = Image.open('.dataset/My_dataset/content/lenna.jpg')
    img3 = Image.open('.dataset/My_dataset/content/golden_gate.jpg')
    img4 = Image.open('.dataset/My_dataset/style/picasso_seated_nude_hr.jpg')
    img5 = Image.open('.dataset/My_dataset/style/hosi.jpg')
    img6 = Image.open('.dataset/My_dataset/content/blonde_girl.jpg')
    img7 = Image.open('.dataset/My_dataset/style/contrast_of_forms.jpg')
    img8 = Image.open('.dataset/My_dataset/style/contrast_of_forms_gray.jpg')
    img9 = Image.open('.dataset/My_dataset/style/candy.jpg')

    tensor_c1 = trans(img3).unsqueeze(0)
    tensor_c2 = trans(img2).unsqueeze(0)
    tensor_s1 = trans(img7).unsqueeze(0)
    tensor_s2 = trans(img9).unsqueeze(0)

    out1 = etc_01(tensor_c1, tensor_c2, tensor_s1, tensor_s2)
    b, c, h, w = out1.shape
    show_histc(out1.view(-1, 2, c, h, w))
    # out2 = etc_02(tensor_c1, tensor_c2, tensor_s1, tensor_s2)
    # out3 = etc_03(tensor_c1, tensor_c2, tensor_s1, tensor_s2)
    # show_sample(out1, tensor_c1, tensor_c2, tensor_s1, tensor_s2, title='Image Color histogram matching Aia')
    # show_sample(out2, tensor_c1, tensor_c2, tensor_s1, tensor_s2, title='Image Color histogram matching Rein')
    # show_sample(out3, tensor_c1, tensor_c2, tensor_s1, tensor_s2, title='Image Color histogram matching Aia-Rein')

    # out4 = etc_04(tensor_c1, tensor_c2, tensor_s1, tensor_s2)
    # show_sample(out4, tensor_c1, tensor_c2, tensor_s1, tensor_s2, title='Image Transfer Color')
