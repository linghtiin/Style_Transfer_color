import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.models import vgg19

from Source.myModels import dataset

vgg11_pth = '.models/vgg11-bbd30ac9.pth'
vgg19_pth = '.models/vgg19-dcbb9e9d.pth'


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = (content_features - content_mean) / content_std
    normalized_features = style_std * normalized_features + style_mean
    return normalized_features


def feat_norm(features):
    feat_mean, feat_std = calc_mean_std(features)
    normalized_features = (features - feat_mean) / feat_std
    return normalized_features


class Block(nn.Module):
    def __init__(self, in_clannel, out_clannel, kernel_size=3, pad_size=1, activate_func=None, activated=True):
        super(Block, self).__init__()
        if activate_func is None:
            activate_func = F.relu
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_clannel, out_clannel, kernel_size)
        self.activate_func = activate_func
        self.activated = activated

    def forward(self, x):
        y = self.pad(x)
        y = self.conv(y)
        if self.activated:
            return self.activate_func(y)
        else:
            return y


class Enc_vgg19(nn.Module):
    def __init__(self):
        super(Enc_vgg19, self).__init__()

        vgg = vgg19()
        pre_dict = vgg19_pth
        vgg.load_state_dict(th.load(pre_dict))
        vgg = vgg.features
        vgg[4] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        vgg[9] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        vgg[18] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        vgg[27] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=True):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class Enc_vgg19_noAvg(Enc_vgg19):
    def __init__(self):
        super(Enc_vgg19_noAvg, self).__init__()

        vgg = vgg19()
        pre_dict = vgg19_pth
        vgg.load_state_dict(th.load(pre_dict))
        vgg = vgg.features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = Block(512, 256, 3, 1)
        self.rc2 = Block(256, 256, 3, 1)
        self.rc3 = Block(256, 256, 3, 1)
        self.rc4 = Block(256, 256, 3, 1)
        self.rc5 = Block(256, 128, 3, 1)
        self.rc6 = Block(128, 128, 3, 1)
        self.rc7 = Block(128, 64, 3, 1)
        self.rc8 = Block(64, 64, 3, 1)
        self.rc9 = Block(64, 3, 3, 1, activated=False)
        for k, v in self.named_parameters():
            if k.find("bias") >= 0:
                init.constant_(v.data, 0)  # bias 初始化为0
            else:
                init.xavier_normal_(v.data)  # 没有预训练，则使用xavier初始化

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h


class Decoder_2p(Decoder):
    def __init__(self):
        super(Decoder_2p, self).__init__()
        self.rc1 = nn.Sequential(Block(1024, 512, 3, 1, activate_func=F.rrelu),
                                 Block(512, 256, 3, 1, activate_func=F.rrelu))

        for k, v in self.named_parameters():
            if k.find("bias") >= 0:
                init.constant_(v.data, 0)  # bias 初始化为0
            else:
                init.xavier_normal_(v.data)  # 没有预训练，则使用xavier初始化


class Pre_Model(nn.Module):
    def __init__(self, dev='cpu'):
        super(Pre_Model, self).__init__()
        self.device = dev
        self.noAdaIn = False
        self.isProColor = True

    @staticmethod
    def calc_content_loss(out_features, target_feature, has_norm=False):
        if has_norm:
            return F.mse_loss(feat_norm(out_features), feat_norm(target_feature))
        else:
            return F.mse_loss(out_features, target_feature)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features, lams):
        loss = 0
        for i, (c, s) in enumerate(zip(content_middle_features, style_middle_features)):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += lams[i] * F.mse_loss(c_mean, s_mean) + lams[i] * F.mse_loss(c_std, s_std)
        return loss

    @staticmethod
    def calc_style_loss2(content_middle_features, style_middle_features, lams):
        loss = 0
        for layer, (c, s) in enumerate(zip(content_middle_features, style_middle_features)):
            batch, channel, h, w = c.shape
            norm_size = channel * h * w / lams[layer]
            _loss = 0
            for i in range(batch):
                view_c = c[i].view(channel, -1)
                gram_c = th.mm(view_c, view_c.t())
                view_s = s[i].view(channel, -1)
                gram_s = th.mm(view_s, view_s.t())
                _loss += F.mse_loss(gram_c.div(norm_size), gram_s.div(norm_size))
            loss += _loss
        return loss

    @staticmethod
    def calc_color_loss(out_images, target_images, lams):
        out_Y, out_I, out_Q = dataset.trans_rgb2yiq(out_images)
        content_Y, content_I, content_Q = dataset.trans_rgb2yiq(target_images)
        loss_color = lams[-2] * F.mse_loss(out_I, content_I) + lams[-1] * F.mse_loss(out_Q, content_Q)
        return loss_color

    @staticmethod
    def calc_tv_loss(images):
        _yl = images[:, :, 1:, :]
        _yr = images[:, :, :-1, :]
        _xl = images[:, :, :, 1:]
        _xr = images[:, :, :, :-1]
        loss = F.mse_loss(_yr, _yl) + F.mse_loss(_xr, _xl)
        return loss


# Model original vgg_19 & decoder
# Nothing to Edit
class Model(Pre_Model):
    def __init__(self, noAdaIN=False):
        super().__init__()
        self.vgg_encoder = Enc_vgg19_noAvg()
        self.decoder = Decoder()
        self.noAdaIn = noAdaIN

    @staticmethod
    def img_norm(images):
        _images = []
        for img in images:
            img_t = dataset.norm(img)
            _images.append(img_t)
        return th.stack(_images)

    def generate(self, content_images, style_images, alpha=1.0, isTrain=False):
        content_images = self.img_norm(content_images)
        style_images = self.img_norm(style_images)
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        if not self.noAdaIn:
            t = adain(content_features, style_features)
            t = alpha * t + (1 - alpha) * content_features
        else:
            t = content_features
        out = self.decoder(t)
        out = dataset.denorm(out, self.device)
        if isTrain:
            return out, t
        else:
            return out

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        out, t = self.generate(content_images, style_images, alpha, isTrain=True)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        loss = self.calc_content_loss(output_features, t)

        if not self.noAdaIn:
            if type(lam) is int:
                lam = (1, 1, 1, 1) * lam
            output_middle_features = self.vgg_encoder(out, output_last_feature=False)
            style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)
            loss_s = self.calc_style_loss(output_middle_features, style_middle_features, lam)
            loss += loss_s

        return loss


class Model8_color1(Pre_Model):
    """
    Color histogram matching Model

    """

    def __init__(self, noAdaIN=False, isProColor=True):
        super(Model8_color1, self).__init__()
        self.vgg_encoder = Enc_vgg19()
        self.decoder = Decoder()
        self.noAdaIn = noAdaIN
        self.isProColor = isProColor

    @staticmethod
    def reper_color(content_image, style_image):
        out = dataset.color_transfer_reinhard(content_image, style_image)
        out = dataset.color_transfer(content_image, out)
        return out

    def generate(self, content_images, style_images, alpha=1.0, _istraining=False):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        new_style = self.reper_color(content_images, style_images)
        style_features = self.vgg_encoder(new_style, output_last_feature=True)

        if not self.noAdaIn:
            t = adain(content_features, style_features)
            t = alpha * t + (1 - alpha) * content_features
        else:
            t = content_features

        out = self.decoder(t)
        out = dataset.images_camp(out)
        if _istraining:
            return out, t
        else:
            return out

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        if type(lam) is int or type(lam) is float:
            lam = (1, 1, 1, 1, 1, 1) * lam

        new_style = self.reper_color(content_images, style_images)
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(new_style, output_last_feature=True)

        if not self.noAdaIn:
            t = adain(content_features, style_features)
            t = alpha * t + (1 - alpha) * content_features
        else:
            t = content_features

        out = self.decoder(t)
        out = dataset.images_camp(out)
        output_features = self.vgg_encoder(out, output_last_feature=True)

        loss = self.calc_content_loss(output_features, t)
        if not self.noAdaIn:
            output_middle_features = self.vgg_encoder(out, output_last_feature=False)
            style_middle_features = self.vgg_encoder(new_style, output_last_feature=False)
            loss_s = self.calc_style_loss(output_middle_features, style_middle_features, lam)
            loss += loss_s

        if self.isProColor:
            loss += self.calc_color_loss(out, content_images, lam)

        return loss


class Model9_color2(Pre_Model):
    """
    Luminance-only transfer Model

    """

    def __init__(self, noAdaIN=False, isProColor=True):
        super(Model9_color2, self).__init__()
        self.vgg_encoder = Enc_vgg19()
        self.decoder = Decoder()
        self.noAdaIn = noAdaIN
        self.isProColor = isProColor

    @staticmethod
    def img_pretrans(images):
        img_y, img_i, img_q = dataset.trans_rgb2yiq(images)
        # img_y = (img_y - 0.485) / 0.225
        return img_y, img_i, img_q

    @staticmethod
    def img_endtrans(images_y, images_i, images_q):
        # images_y = images_y * 0.225 + 0.485
        images = dataset.trans_yiq2rgb(images_y, images_i, images_q)
        return images

    def generate(self, content_images, style_images, alpha=1.0, _istraining=False):
        content_Y, content_I, content_Q = self.img_pretrans(content_images)
        style_Y, style_I, style_Q = self.img_pretrans(style_images)
        content_Ys = content_Y.unsqueeze(1).expand(-1, 3, -1, -1)
        style_Ys = style_Y.unsqueeze(1).expand(-1, 3, -1, -1)
        content_Ys_features = self.vgg_encoder(content_Ys, output_last_feature=True)
        style_Ys_features = self.vgg_encoder(style_Ys, output_last_feature=True)

        if not self.noAdaIn:
            t = adain(content_Ys_features, style_Ys_features)
            t = alpha * t + (1 - alpha) * content_Ys_features
        else:
            t = content_Ys_features
        out_Y = self.decoder(t).mean(dim=1)
        out = self.img_endtrans(out_Y, content_I, content_Q)

        if _istraining:
            return out, t
        else:
            return out

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        if type(lam) is int or type(lam) is float:
            lam = (1, 1, 1, 1, 1, 1) * lam

        out, t = self.generate(content_images, style_images, alpha, _istraining=True)
        output_features = self.vgg_encoder(out, output_last_feature=True)

        loss = self.calc_content_loss(output_features, t)
        if not self.noAdaIn:
            output_middle_features = self.vgg_encoder(out, output_last_feature=False)
            style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)
            loss_s = self.calc_style_loss(output_middle_features, style_middle_features, lam)
            loss += loss_s

        if self.isProColor:
            loss_color = self.calc_color_loss(out, content_images, lam)
            loss += loss_color

        return loss


class Model16_color4(Pre_Model):
    def __init__(self, noAdaIN=False, isProColor=True):
        super(Model16_color4, self).__init__()
        self.vgg_encoder = Enc_vgg19()
        self.decoder = Decoder_2p()
        self.noAdaIn = noAdaIN
        self.isProColor = isProColor

    @staticmethod
    def reper_color(content_image, style_image):
        out = dataset.color_transfer_reinhard(content_image, style_image)
        out = dataset.color_transfer(content_image, out)
        return out

    @staticmethod
    def get_img_color(images):
        img_yiq = dataset.trans_rgb2yiq(images)
        img_color = dataset.trans_yiq2rgb(th.zeros_like(img_yiq[0]), img_yiq[1], img_yiq[2])
        return img_color

    def generate(self, content_images, style_images, alpha=1.0):
        style_new = self.reper_color(content_images, style_images)
        imgc_color = self.get_img_color(content_images)
        content_features = self.vgg_encoder(content_images)
        content_feature_color = self.vgg_encoder(imgc_color)
        style_features = self.vgg_encoder(style_new)

        if not self.noAdaIn:
            t = adain(content_features, style_features)
            t = alpha * t + (1 - alpha) * content_features
        else:
            t = content_features
        out = self.decoder(th.cat([t, content_feature_color], dim=1))
        out = dataset.images_camp(out)
        return out

    def forward(self, content, style, alpha=1.0, lam=10):
        if type(lam) is int or type(lam) is float:
            lam = [i * lam for i in (1, 1, 1, 1, 1, 1)]
        out = self.generate(content, style, alpha)

        content_features = self.vgg_encoder(content)
        out_middle_features = self.vgg_encoder(out, output_last_feature=False)
        loss = self.calc_content_loss(out_middle_features[-1], content_features, has_norm=True)

        if not self.noAdaIn:
            style_new = self.reper_color(content, style)
            style_middle_features = self.vgg_encoder(style_new, output_last_feature=False)
            loss_s = self.calc_style_loss(out_middle_features, style_middle_features, lam)
            loss += loss_s

        if self.isProColor:
            loss_color = self.calc_color_loss(out, content, lam)
            loss += loss_color

        return loss


if __name__ == '__main__':
    import os
    from PIL import Image
    from Source.myModels.dataset import trans, norm, denorm
    from Source.myTools.TrainTools import show_tensor

    os.chdir('G:\\Program\\Git\\My_Git\\test\\Grand_work\\exam_01_VGG_19')

    device = th.device('cuda:0')

    m1 = Model().to(device=device)
    m1.device = device
    m1.load_state_dict(th.load('.models\\pre_model_state.pth', map_location=device))
    m1.eval()

    img1 = Image.open('.dataset\\My_dataset\\test-1.bmp')
    img2 = Image.open('.dataset\\My_dataset\\style\\brushstrokes.jpg')
    img3 = Image.open('.dataset\\My_dataset\\content\\golden_gate.jpg')
    img4 = Image.open('.dataset\\style_train\\209.jpg')

    t1 = trans(img3).unsqueeze(0).to(device)
    t2 = trans(img4).unsqueeze(0).to(device)
    show_tensor(t1)
    show_tensor(t2)

    o = m1.generate(t1, t2)
    show_tensor(o.detach())
    lt = m1(t1, t2)
    print(lt.detach().cpu().numpy())
