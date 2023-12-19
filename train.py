import os
import sys
import time
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
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import models


from PIL import Image
import logging

from utils import StyleTranslator, StyleTranslatorSharedDataset, epoch, ParamDiffAug, DiffAugment
from vit2 import ViT


def config_logging(
    rank=0, 
    verbose=False, 
    append=False
):
    log_file = f'out_{rank}.log'
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if log_file is not None:
        file_mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

def load_styles(file_path, num_translators=5):
    weights_dict = torch.load(file_path, map_location=torch.device('cpu'))
    translators = [StyleTranslator() for _ in range(num_translators)]
    for i, translator in enumerate(translators):
        translator_keys = {k.split('.', 1)[1]: v for k, v in weights_dict.items() if k.startswith(f'{i}.')}
        translator.load_state_dict(translator_keys)
    translators = nn.ModuleList(translators)
    return translators

def load_distilled_cifar10_images(file_path):
    images = torch.load(file_path, map_location=torch.device("cuda:0"))
    return images

def load_test(transform):
    if args.use_cifar_10:
        test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True, transform=transform)
    elif args.use_cifar_100:
        test_set = torchvision.datasets.CIFAR100(root="./datasets", train=False, download=True, transform=transform)
    test_load = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_train,
        shuffle=False,
        num_workers=2
    )
    return test_load

def test(testloader, model):
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in testloader:
            batch = [t.to(args.device) for t in batch]
            images, labels = batch
            logits, _ = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
    accuracy = correct / len(testloader.dataset)
    return accuracy

class PreprocessedDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def init_distributed_nodes():
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    # dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank)
    dist.init_process_group(backend='nccl')
    return rank, n_ranks

def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass

def build_vit():
    model = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 8,
        depth = 4,
        heads = 2,
        mlp_dim = 4,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    return model
    
def main(args):
    
    rank = 0

    if args.distributed_mode:
        rank, n_ranks = init_distributed_nodes()
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    config_logging(rank=rank)
    logging.info("Training start!")
    
    save_dir = 'preprocessed_dataset'        
    
    if args.distributed_mode:
        args.device = torch.device("cuda", local_rank)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.dsa_param = ParamDiffAug()
    
    try_barrier()
    
    is_transformer = True
    if is_transformer:
        transform = ToTensor()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize the images
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),  # ResNet typically requires 224x224 input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    """ Dataset Preparation """
    if args.use_cifar_10 and not args.use_distilled_set:
        print("use cifar 10")
        trainset = torchvision.datasets.CIFAR10(root="./datasets", train=True,
                                            download=True, transform=transform)
    elif args.use_cifar_100 and not args.use_distilled_set:
        print("use cifar 100")
        trainset = torchvision.datasets.CIFAR100(root="./datasets", train=True,
                                            download=True, transform=transform)
    else:
        print("use preprocessed dataset")
        trainset = PreprocessedDataset(save_dir, transform=transform)

    sampler = None
    
    if args.distributed_mode:
        sampler = DistributedSampler(trainset)

    train_loader = DataLoader(trainset, batch_size=32, sampler=sampler, num_workers=2)
    
    # args.lr_net = torch.tensor(args.lr_teacher).item()
    test_loader = load_test(transform)
    
    """ Model Preparation """
    if is_transformer:
        model = build_vit()
    else:
        model = models.resnet18(pretrained=False)  # Using a pre-trained ResNet18
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        
    model = model.to(device)
    
    if args.distributed_mode:
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    logging.info(f"distribute mode {args.distributed_mode}")
    logging.info(f"CUDNN STATUS: {torch.backends.cudnn.enabled}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    """ Training and Validation """
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss().to(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 200
    for epoch in range(num_epochs):
        logging.info(f"Start epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        end_time = time.time()
        duration = end_time - start_time

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        print(f"Start epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        logging.info(f"Epoch {epoch+1} training_seconds: {duration}")
    
    logging.info("Training Finished!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--local_rank', type=int, default=0, help='Rank of the node for multi-gpu distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of the node for multi-node distributed training')
    parser.add_argument('--gpus', type=int, default=4, help='Total number of gpus per node')
    # ... other arguments ...

    parser.add_argument('--use_cifar_10', type=bool, default=False, help='Use cifar 10 as training set')
    parser.add_argument('--use_cifar_100', type=bool, default=False, help='Use cifar 100 as training set')
    parser.add_argument('--use_distilled_set', type=bool, default=False, help='Use distilled dataset as training set')
    
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

    # parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    # parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    # parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    # parser.add_argument('--load_all', action='store_true',
    #                     help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

#     parser.add_argument('--max_files', type=int, default=None,
#                         help='number of expert files to read (leave as None unless doing ablations)')
#     parser.add_argument('--max_experts', type=int, default=None,
#                         help='number of experts to read per file (leave as None unless doing ablations)')

#     parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--n_style', type=int, default=5, help='the number of styles')
    # parser.add_argument('--single_channel', action='store_true', help="using single-channel but more basis")
    # parser.add_argument('--lambda_club_content', type=float, default=0.1)
    # parser.add_argument('--lambda_likeli_content', type=float, default=1.)
    # parser.add_argument('--lambda_cls_content', type=float, default=1.)
    # parser.add_argument('--lambda_contrast_content', type=float, default=1.)

    args = parser.parse_args()

    main(args)
    
    

    
#     # Initialize communication profiler
#     world_size = dist.get_world_size()
#     gpus_per_node = 4  # Change this based on your setup
#     profiler = CommunicationProfiler(world_size, gpus_per_node)

#     # Example usage in a DDP training loop
#     for data in dataloader:
#         optimizer.zero_grad()
#         output = model(data)

#         loss = output.loss()
#         loss.backward()

#         # Determine if the communication is intra-node or inter-node
#         is_intra_node = profiler.is_intra_node_comm(target_rank)

#         # Start communication profiling
#         profiler.communication_start(is_intra_node)
#         dist.all_reduce(loss)  # Example communication
#         # End communication profiling
#         profiler.communication_end(size_of(loss))

#         optimizer.step()


    # # At the end of training
    # intra_node_data, inter_node_data, intra_node_comm_time, inter_node_comm_time = profiler.get_stats()
    # print(f"Intra-node Data Transferred: {intra_node_data} bytes, Intra-node Communication Time: {intra_node_comm_time} seconds")
    # print(f"Inter-node Data Transferred: {inter_node_data} bytes, Inter-node Communication Time: {inter_node_comm_time} seconds")
    