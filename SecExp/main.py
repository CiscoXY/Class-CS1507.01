import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.utils.tensorboard as tensorboard

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0





## 更改开始的地方:
LabelList = os.listdir(r'tiny_imagenet\train') # 获取train集下面的文件夹名list
f = open('tiny_imagenet\\val' + '\\val_annotations.txt' , 'r') # 打开文件
txt = []
for line in f:
    txt.append(line.split())  # 获得图片的信息组成的list
f.close()
new_label_mapping = []
for index , text in enumerate(txt):
    new_label_mapping.append([ text[0], LabelList.index(text[1])]) #获得val图片名称所对应的train文件夹名称所对应的index，也就是需要替换的标签，这是一个二维list
new_label_mapping = dict(new_label_mapping) # 构建新的映射，也就是字典
    
def target_transformfunction(target):
    return new_label_mapping[target]


class valImageFolder(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            Original_name = False , # 这个参数主要用于找出判断不同的照片，为的是任务5，默认为不显示，也就是只返回映射后的标签
    ):
        super(valImageFolder, self).__init__(root, transform, target_transform) # 根据下面的具体，我们只需要前三个参数就够了
        self.imgs = self.samples
        self.Original_name = Original_name
    def __getitem__(self, index: int) -> Tuple[Any, Any]: # 覆写一下__getitem__这个方法
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = path[25:] #target调整为这个图片的名称
        if (self.Original_name == True):
            picname = target
        else:
            picname = None
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target= self.target_transform(target)
        return sample, target , picname# 如果这么参数为Ture,那么picname不是None，而是这个文件的文件名
## 更改完成
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        fc_features = model.fc.in_features   #提取最后一层的参数
        model.fc = nn.Linear(fc_features, 200) # 给他更改成200个类别

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val') # 这俩东西是个path,=
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(  # 这是一系列可迭代的数据对象
        traindir, #打进来一个path
        transforms.Compose([  #对这个path里面的数据进行的操作
            #transforms.RandomResizedCrop(224), 不需要随机裁剪
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = valImageFolder(valdir, transforms.Compose([ # 我也给他弄成一个可迭代的对象
            #transforms.Resize(256), 不需要随机裁剪和尺寸转化
            #transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize,
        ]) , target_transform=target_transformfunction)
    
    val_loader = torch.utils.data.DataLoader(       #验证的数据集 结构跟训练数据集是一样的
        val_dataset,
        batch_size=args.batch_size, shuffle=False,  #批训练数据个数=256
        num_workers=args.workers, pin_memory=True)
    
    if args.evaluate:
        val_dataset = valImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]) , target_transform=target_transformfunction , Original_name = True)
        val_loader = torch.utils.data.DataLoader(       #验证的数据集 结构跟训练数据集是一样的
        val_dataset,
        batch_size=args.batch_size, shuffle=False,  #批训练数据个数=256
        num_workers=args.workers, pin_memory=True)
        validate(val_loader, model, criterion, args ,None , None , different = True)
        return
    
    #TODO 生成一张假的图片，输入到网络去看效果
    #images = torch.randn(64, 3, 7, 7)
    #graphwriter = tensorboard.SummaryWriter(comment = '_graph') # 绘制结构的writer
    #graphwriter.add_graph(model , input_to_model = images)
    #graphwriter.close()   # 关掉这个writer，写一张图就够了
    #TODO 建立横轴为epoch，纵轴为准确率或者loss的writer
    epoch_paramwriter = tensorboard.SummaryWriter(log_dir = 'runs/total_epoch_information') # 观察loss，精准度等等的参数,跨度为1个epoch记录一次
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args , epoch_paramwriter) # 增加了一个参数，writer，方便传入上面的epoch_paramwriter

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args , epoch_paramwriter , epoch)# 增加了2个参数，writer，方便传入上面的epoch_paramwriter , epoch,方便记录
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
    epoch_paramwriter.close()
    

def train(train_loader, model, criterion, optimizer, epoch, args , writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    in_one_epoch_parameterwriter = tensorboard.SummaryWriter(log_dir='runs/epoch'+str(epoch)+'_parameters') #对于不同的epoch，绘制针对这个epoch的图像
    
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        in_one_epoch_parameterwriter.add_scalar('losses/current_loss', losses.val, i) # 绘制点(i , losses.val)
        in_one_epoch_parameterwriter.add_scalar('losses/average_loss_tillnow', losses.avg, i)# 绘制点(i , losses.avarage)
        in_one_epoch_parameterwriter.add_scalar('accuracy/top1_current', top1.val, i)
        in_one_epoch_parameterwriter.add_scalar('accuracy/top1_average', top1.avg, i)
        in_one_epoch_parameterwriter.add_scalar('accuracy/top5_current', top5.val, i)
        in_one_epoch_parameterwriter.add_scalar('accuracy/top5_average', top5.avg, i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    in_one_epoch_parameterwriter.close()# 每个epoch单独绘制一个，所以要关闭
    
    writer.add_scalar('train/losses/current_epoch' , losses.val , epoch)
    writer.add_scalar('train/losses/avarage_epoch_tillnow' , losses.avg , epoch)
    writer.add_scalar('train/accuracy/top1_current_epoch', top1.val, epoch)
    writer.add_scalar('train/accuracy/top1_average_epoch', top1.avg, epoch)
    writer.add_scalar('train/accuracy/top5_current_epoch', top5.val, epoch)
    writer.add_scalar('train/accuracy/top5_average_epoch', top5.avg, epoch)

def validate(val_loader, model, criterion, args , writer , epoch , different = False): # 默认不找不同，如果需要找不同，那么就是任务5
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        count = 0 # 到3就截止，因为只需要找10张就可以，我们最多最多需要3*256张
        for i, (images, target , picname) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            
            
            if(different and count == 0):
                _, pred = output.topk(5, 1, True, True)
                pred = pred.t() # 转置了一下
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                correct = correct.t() # 再转置回来
                count += 1 # 找到一张了
                print(picname) # 如果找不同那个需要是True，那么如果output不是target，那么输出这个文件名，以及target和对应的output
                print(target)
                for i in correct:
                    print(i)  # 上面一堆是打印出来这些判断，认为进行挑选
            
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time() 

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
    if(writer is not None and epoch is not None):
        writer.add_scalar('val/losses/current_epoch' , losses.val , epoch)
        writer.add_scalar('val/losses/avarage_epoch_tillnow' , losses.avg , epoch)
        writer.add_scalar('val/accuracy/top1_current_epoch', top1.val, epoch)
        writer.add_scalar('val/accuracy/top1_average_epoch', top1.avg, epoch)
        writer.add_scalar('val/accuracy/top5_current_epoch', top5.val, epoch)
        writer.add_scalar('val/accuracy/top5_average_epoch', top5.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint1.pth.tar'):  #TODO 到时候记得改一下checkpoint的名字
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    if state['epoch']%4 == 0:
        torch.save(state , 'checkpoint' + '_epoch' + str(state['epoch']) + '.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})' #所以第一个是当前的value，括号里面的是平均值
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # 获得前maxk个类别得分最高的项，也就是batch_size x maxk的矩阵，每一行的数值降序排列 , 每个数值就是原始数据第k大的数值对应的的index值
        pred = pred.t() # 转置了一下
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # 得到的预测是否和真实值相等，形状为maxk x batch_size 其中ij元素表示第j个训练元的第i个相像的结果是否是真实值
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()