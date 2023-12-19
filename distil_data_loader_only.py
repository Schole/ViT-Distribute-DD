import os
import argparse
import wandb
import torch
import torchvision
import torch.nn.functional as F
import torchvision.utils
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToPILImage
from PIL import Image

# from utils.logging import config_logging
# from utils.distributed import init_workers, try_barrier

from utils import StyleTranslator, StyleTranslatorSharedDataset, epoch, ParamDiffAug, DiffAugment
from vit import ViTForClassfication


def load_and_inspect(file_path):
    data = torch.load(file_path, map_location=torch.device('cpu'))

    print("Type of loaded data:", type(data))
    if isinstance(data, torch.Tensor):
        print("Shape of the tensor:", data.shape)
    elif isinstance(data, dict):
        print("Keys in the dictionary:", data.keys())
    else:
        print("Data is neither a tensor nor a dictionary.")
        
    return data

def load_styles(file_path, num_translators=5):
    weights_dict = torch.load(file_path, map_location=torch.device("cuda:0"))
    translators = [StyleTranslator() for _ in range(num_translators)]
    
    for i, translator in enumerate(translators):
        translator_keys = {k.split('.', 1)[1]: v for k, v in weights_dict.items() if k.startswith(f'{i}.')}
        translator.load_state_dict(translator_keys)
    translators = nn.ModuleList(translators)
    
    if torch.cuda.is_available():
        # Move each model in the ModuleList to GPU
        for i, model in enumerate(translators):
            translators[i] = model.to('cuda')
    
    return translators

def load_distilled_cifar10_images(file_path):
    images = torch.load(file_path, map_location=torch.device("cuda:0"))
    
    return images

    
def main(args):
    """ Initialization """
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args.dsa_param = ParamDiffAug()
    
    images_path = "images_best.pt" 
    distilled_images = load_distilled_cifar10_images(images_path)
    # print(distilled_images.shape)
    
    model_path = "weights_best.pt"
    loaded_models = load_styles(model_path)
    print(f"translators: {loaded_models}")

    dst_train = StyleTranslatorSharedDataset(distilled_images, loaded_models)
    to_pil_image = ToPILImage()

    # Create a directory to save images
    save_dir = 'preprocessed_dataset'
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over the dataset and save images
    for i in range(len(dst_train)):
        print(f'Process image {i}')
        image, label = dst_train[i]
        image_pil = to_pil_image(image)  # Convert to PIL Image if not already
        image_pil.save(os.path.join(save_dir, f'image_{i:05d}_{label}.png'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of the node for multi-node distributed training')
    parser.add_argument('--gpus', type=int, default=4, help='Total number of gpus per node')
    # ... other arguments ...
    
    parser.add_argument('-d', '--distributed-mode', type=bool, default=False)
        
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')


    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=1000, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=500,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=10000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=100, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_style', type=float, default=100, help='learning rate for updating style translator')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true',
                        help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--max_files', type=int, default=None,
                        help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None,
                        help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--n_style', type=int, default=5, help='the number of styles')
    parser.add_argument('--single_channel', action='store_true', help="using single-channel but more basis")
    parser.add_argument('--lambda_club_content', type=float, default=0.1)
    parser.add_argument('--lambda_likeli_content', type=float, default=1.)
    parser.add_argument('--lambda_cls_content', type=float, default=1.)
    parser.add_argument('--lambda_contrast_content', type=float, default=1.)

    args = parser.parse_args()

    main(args)
    
    
    
    