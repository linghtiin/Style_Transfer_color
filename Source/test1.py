"""
    单图像模型测试

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
from Source.myTools.TrainTools import check_gpu, load_model

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))

trans = transforms.Compose([transforms.ToTensor(),
                            # normalize,
                            ])


def denorm(tensor, device):
    std = th.tensor((0.485, 0.456, 0.406)).reshape(-1, 1, 1).to(device)
    mean = th.tensor((0.229, 0.224, 0.225)).reshape(-1, 1, 1).to(device)
    res = th.clamp(tensor * std + mean, 0, 1)
    return res


def test(args):
    device = check_gpu(args.gpu)

    model = load_model(args, device)

    model.eval()

    c = Image.open(args.content)
    s = Image.open(args.style)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with th.no_grad():
        out = model.generate(c_tensor, s_tensor, alpha=args.alpha)

    if args.output_name is None:
        c_name = Path(args.content).stem
        s_name = Path(args.style).stem
        args.output_name = f'{c_name}_{s_name}'

    save_image(out, f'{args.output_name}.jpg', nrow=1)
    o = Image.open(f'{args.output_name}.jpg')

    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)
    s = s.resize((i // 4 for i in c.size))

    demo.paste(c, (0, 0))
    demo.paste(o, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)

    o.paste(s, (0, o.height - s.height))
    o.save(f'{args.output_name}_with_style_image.jpg', quality=95)

    print(f'result saved into files starting with {args.output_name}')


class Parser:
    def __init__(self):
        self.content = None
        self.style = None
        self.output_name = None
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
    # cmd_args.alpha = 0.9
    # cmd_args.gpu = -1

    cmd_args.content = '.dataset/My_dataset/content/golden_gate.jpg'
    # cmd_args.content = '.dataset/My_dataset/content/blonde_girl.jpg'

    # cmd_args.style = '.dataset/My_dataset/style/contrast_of_forms.jpg'
    # cmd_args.style = '.dataset/My_dataset/style/contrast_of_forms_gray.jpg'
    # cmd_args.style = '.dataset/My_dataset/style/en_campo_gris.jpg'
    cmd_args.style = '.dataset/My_dataset/style/hosi.jpg'
    # cmd_args.style = '.dataset/My_dataset/style/brushstrokes.jpg'
    # cmd_args.style = '.dataset/style_train/255.jpg'
    # cmd_args.style = '.dataset\My_dataset\style\candy.jpg'

    # cmd_args.model = Models.Model16_color4
    # cmd_args.reuse = '.models/pre_model_color4.pth'

    # cmd_args.output_name = 'golden_gate_209_after'
    test(cmd_args)
