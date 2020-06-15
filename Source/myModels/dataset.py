import numpy as np
import torch as th
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as th_trans
from PIL import Image
import colorsys
import cv2 as cv

my_format = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

trans = th_trans.Compose([th_trans.RandomCrop(256),
                          th_trans.ToTensor(),
                          # th_trans.Normalize(mean=my_format[0],
                          #                    std=my_format[1])
                          ])


def trans_rgb2yiq(images):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    image_Y = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    image_I = 0.596 * images[:, 0] - 0.274 * images[:, 1] - 0.322 * images[:, 2]
    image_Q = 0.211 * images[:, 0] - 0.522 * images[:, 1] + 0.311 * images[:, 2]
    return image_Y, image_I, image_Q


def trans_yiq2rgb(img_Y, img_I, img_Q):
    image_R = th.clamp(img_Y + 0.956 * img_I + 0.620 * img_Q, 0, 1)
    image_G = th.clamp(img_Y - 0.272 * img_I - 0.648 * img_Q, 0, 1)
    image_B = th.clamp(img_Y - 1.105 * img_I + 1.705 * img_Q, 0, 1)
    return th.stack([image_R, image_G, image_B], dim=1)


def norm(tensor):
    normalize = th_trans.Normalize(mean=my_format[0],
                                   std=my_format[1])
    return normalize(tensor)


def denorm(tensor, device):
    std = th.tensor(my_format[1]).reshape(-1, 1, 1).to(device)
    mean = th.tensor(my_format[0]).reshape(-1, 1, 1).to(device)
    res = th.clamp(tensor * std + mean, 0, 1)
    return res


def images_camp(images):
    b, c, h, w = images.shape
    Imax, _ = th.max(images.view(b, c, -1), dim=2)
    Imin, _ = th.min(images.view(b, c, -1), dim=2)
    out = (images - Imin.view(b, c, 1, 1)) / (Imax.view(b, c, 1, 1) - Imin.view(b, c, 1, 1))
    return out


def color_cov_sqrt(images):
    b, c, h, w = images.shape
    img_mean = th.mean(images, (2, 3))
    img_temp1 = images.view([b, c, 1, -1]) - img_mean.view([b, c, 1, 1])
    img_temp2 = th.matmul(img_temp1.permute(0, 3, 1, 2), img_temp1.permute(0, 3, 2, 1))
    img_cov = th.sum(img_temp2, dim=1) / (h * w)
    img_cov_sqrt = []
    for cov in img_cov:
        img_cov_eig, img_cov_eig_vector = cov.eig(eigenvectors=True)
        cov_sqrt = th.mm(th.mm(img_cov_eig_vector, th.diag_embed(th.pow(img_cov_eig[:, 0], 0.5))),
                         img_cov_eig_vector.permute(1, 0))
        img_cov_sqrt.append(cov_sqrt)
    return th.stack(img_cov_sqrt)


def color_transfer(content, style):
    """
    To match images`s color in Image Analogies
    If can`t, return style images originally.
    :param content: a tensor for match color, with BxCxHxW
    :param style: a tensor for style images, with BxCxHxW
    :return: a tensor with BxCxHxW
    """
    b, c, h, w = style.shape
    content_cq = color_cov_sqrt(content)
    style_cq = color_cov_sqrt(style)

    _skip_list = set()
    _sample_fin = th.isfinite(th.stack([content_cq, style_cq], dim=1))
    _sample_det = style_cq.det().abs() < 1e-9
    for ib in range(b):
        if not _sample_fin[ib].all():
            _skip_list.add(ib)
        if _sample_det[ib]:
            style_cq[ib] = th.eye(3)
            _skip_list.add(ib)

    mat_Aia = th.bmm(content_cq, style_cq.inverse())
    vator_b = th.mean(content, (2, 3)) - th.bmm(mat_Aia, th.mean(style, (2, 3)).unsqueeze(-1)).squeeze(-1)

    out_images = []
    for ib in range(b):
        if ib in _skip_list:
            out_images.append(style[ib])
        else:
            out_img = (th.matmul(mat_Aia[ib], style[ib].permute(1, 2, 0).unsqueeze(-1)).squeeze(-1)
                       + vator_b[ib].view(1, 1, c)).permute(2, 0, 1)
            out_images.append(out_img)
    out_images = images_camp(th.stack(out_images, dim=0))
    return out_images


def color_channel_cap(content_channel, style_channel):
    b, _, _ = content_channel.shape
    c_mean = th.mean(content_channel, (1, 2)).view(-1, 1, 1)
    s_mean = th.mean(style_channel, (1, 2)).view(-1, 1, 1)
    c_std = th.std(content_channel, (1, 2)).view(-1, 1, 1)
    s_std = th.std(style_channel, (1, 2)).view(b, 1, 1)
    out_style_channel = c_std / s_std * (style_channel - s_mean) + c_mean
    return out_style_channel


def color_transfer_reinhard(content, style):
    """
    To match images`s color in Reinhard
    :param content: a tensor for match color
    :param style: a tensor for style images
    :return: a tensor with BxCxHxW
    """
    content_yiq = trans_rgb2yiq(content)
    style_yiq = trans_rgb2yiq(style)
    out_yiq = []
    for c, s in zip(content_yiq, style_yiq):
        out_yiq.append(color_channel_cap(c, s))
    out_rgb = trans_yiq2rgb(out_yiq[0], out_yiq[1], out_yiq[2])

    return out_rgb


class PreprocessDataset(Dataset):
    """
    Dataset Object with resized and transform
    """

    def __init__(self, content_dir, style_dir, transforms=trans):
        """
        Dataset initial function.
        :param content_dir: str or Path object
        :param style_dir: str or Path object
        :param transforms: torchvision`s transforms object, defualt: Compose(RandomCrop-256 + ToTensor)
        """
        content_dir_resized = Path(content_dir) / '_resized'
        style_dir_resized = Path(style_dir) / '_resized'
        if not content_dir_resized.exists():
            content_dir_resized.mkdir()
            self._resize(content_dir, content_dir_resized)
        if not style_dir_resized.exists():
            style_dir_resized.mkdir()
            self._resize(style_dir, style_dir_resized)

        content_images = []
        style_images = []
        for s_img in style_dir_resized.glob('*'):
            style_images.append(s_img)
        for c_img in content_dir_resized.glob('*'):
            content_images.append(c_img)

        if len(style_images) * 2 < len(content_images):
            style_images = style_images * 2
        self.image_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(Path(source_dir).glob('*')):
            filename = i.stem + '.bmp'
            try:
                # image = io.imread(str(i))
                image = cv.imread(str(i))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    # image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    image = cv.resize(image, (W, H), interpolation=cv.INTER_CUBIC)
                    # io.imsave(os.path.join(target_dir, filename), image)
                    cv.imwrite(str(Path(target_dir) / filename), image)
            except:
                continue

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        content_img, style_img = self.image_pairs[index]
        content_img = Image.open(content_img)
        style_img = Image.open(style_img)

        if self.transforms:
            content_img = self.transforms(content_img)
            style_img = self.transforms(style_img)
        return content_img, style_img


def test01(tensor):
    tensor_yiq = trans_rgb2yiq(tensor)
    tensor_yiq_re_rgb1 = trans_yiq2rgb(0 * th.ones_like(tensor_yiq[0]), tensor_yiq[1], tensor_yiq[2])
    tensor_yiq_re_rgb2 = trans_yiq2rgb(tensor_yiq[0], th.zeros_like(tensor_yiq[1]), th.zeros_like(tensor_yiq[2]))

    show_tensor(tensor)
    show_tensor(tensor_yiq[0].unsqueeze(1).expand(-1, 3, -1, -1))
    show_tensor(tensor_yiq_re_rgb1)
    show_tensor(tensor_yiq_re_rgb2)


def test02(tensor):
    nt = norm(tensor)
    dn_nt = denorm(nt, 'cpu')

    print(tensor.std((1, 2)), tensor.mean((1, 2)))
    print(dn_nt.std((1, 2)), dn_nt.mean((1, 2)))

    print('nt:\n', nt.std((1, 2)), nt.mean((1, 2)))


def test03(content, style):
    out = color_transfer(content, style)

    show_tensor(content)
    show_tensor(style)
    show_tensor(out)

    return out


if __name__ == '__main__':
    import os
    from Source.myTools.TrainTools import show_tensor

    os.chdir('G:/Program/Git/My_Git/test/Grand_work/exam_01_VGG_19')

    img1 = Image.open('.dataset/My_dataset/test-1.bmp')
    img2 = Image.open('.dataset/My_dataset/content/lenna.jpg')
    img3 = Image.open('.dataset/My_dataset/content/golden_gate.jpg')
    img4 = Image.open('.dataset/My_dataset/style/candy.jpg')
    img5 = Image.open('.dataset/My_dataset/style/Composition-VII.jpg')

    t1 = trans(img1).unsqueeze(0)
    tensor_c1 = trans(img2).unsqueeze(0)
    tensor_c2 = trans(img3).unsqueeze(0)
    tensor_s1 = trans(img4).unsqueeze(0)
    tensor_s2 = trans(img5).unsqueeze(0)

    # test01(tensor_content)
    # test02(tensor_style)
    tensor_out = test03(tensor_c2, tensor_s1)
