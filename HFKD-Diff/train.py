import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops
from lib.utils.utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor,resolve_data_config,adjust_learning_rate
import torchvision.datasets
from timm.data import create_loader
torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    '''distributed'''
    init_dist(args)
    init_logger(args)
    
    # save args
    logger.info(args)
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build dataloader'''
    model_name = args.teacher_model
    data_config = resolve_data_config(vars(args), model=None, verbose=args.rank == 0)
    #timm的预训练模型输入需要[3,224,224],所以需要对cifar100数据集做处理
    if args.dataset == 'cifar100' and model_name.startswith('timm_'):
        train_data = torchvision.datasets.CIFAR100(args.data_path, train=True)
        val_data = torchvision.datasets.CIFAR100(args.data_path, train=False)
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
            
        # data_config['mean'] = (0.5071, 0.4865, 0.4409)
        # data_config['std'] = (0.2673, 0.2564, 0.2762)
        # data_config['interpolation'] = 'bilinear'
        
        train_loader = create_loader(
        train_data,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=0,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=False,
        collate_fn=None,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=False,
        worker_seeding=args.worker_seeding,
        )

        val_loader = create_loader(
            val_data,
            input_size=data_config['input_size'],
            batch_size= args.batch_size,
            is_training=False,
            use_prefetcher=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=False,
            crop_pct=None,
            pin_memory=args.pin_mem,
        )
    else:
        train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)


    '''build model'''
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()
    val_loss_fn = loss_fn

    model = build_model(args,args.model)
    print(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    # Diverse Branch Blocks
    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    model.cuda()
    

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        print(teacher_model)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        teacher_model.cuda()
        # test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        # logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(model, teacher_model, args.model, args.teacher_model, loss_fn, 
                         args.kd, args.ori_loss_weight, args.kd_loss_weight, args.kd_loss_kwargs, 
                         args.assist_kd, args.cross_head,args.student_is_cnn, args)

    # model = DDP(model,
    #             device_ids=[args.local_rank],
    #             find_unused_parameters=False)
    loss_fn.student = model
    logger.info(model)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''build optimizer'''
    optimizer = build_optimizer(args.opt,
                                model,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)

    '''build scheduler'''
    if args.sched != "step":
        steps_per_epoch = len(train_loader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        decay_steps = args.decay_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        scheduler = build_scheduler(args.sched,
                                    optimizer,
                                    warmup_steps,
                                    args.warmup_lr,
                                    decay_steps,
                                    args.decay_rate,
                                    total_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    decay_by_epoch=args.decay_by_epoch,
                                    min_lr=args.min_lr)
    else:
        scheduler=None

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn
        dyrep = DyRep(
            model.module,
            optimizer,
            recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
                                           args.dyrep_recal_bn_iters, m),
            filter_bias_and_bn=not args.opt_no_filter)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'dyrep': dyrep
                                     })

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if args.sched != "step":
            if start_epoch > args.warmup_epochs:
                scheduler.finished = True
            scheduler.step(start_epoch * len(train_loader))
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )
    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None

    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        # train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs
        
        if args.sched == "step":          
            adjust_learning_rate(optimizer, epoch, args)
            
        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader,
                              optimizer, loss_fn, scheduler, auxiliary_buffer,
                              dyrep, loss_scaler)

        # validate
        test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
        if model_ema is not None:
            test_metrics = validate(args,
                                    epoch,
                                    model_ema.module,
                                    val_loader,
                                    val_loss_fn,
                                    log_suffix='(EMA)')

        # dyrep
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    # adjust
                    logger.info('DyRep: adjust model.')
                    dyrep.adjust_model()
                    logger.info(
                        f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
                    )
                    # re-init DDP
                    model = DDP(model.module,
                                device_ids=[args.local_rank],
                                find_unused_parameters=True)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info('DyRep: recalibrate BN.')
                    recal_bn(model.module, train_loader, 200)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics)
        logger.info('\n'.join(['Checkpoints:'] + [
            '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
        ]))


def train_epoch(args,epoch,model,model_ema,loader,optimizer,
                loss_fn,scheduler,auxiliary_buffer=None,dyrep=None,
loss_scaler=None):
    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    batch_time_m = AverageMeter()
    start_time = time.time()
    
    model.train()
    # model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=False) for speedup
        for p in model.parameters():
            p.grad = None
            # p.requires_grad = False#cka

        with torch.cuda.amp.autocast(enabled=loss_scaler is not None):
            if not args.kd:
                output = model(input)
                loss = loss_fn(output, target)
            else:
                loss = loss_fn(input, target)
    
            if auxiliary_buffer is not None:
                loss_aux = loss_fn(auxiliary_buffer.output, target)
                loss += loss_aux * auxiliary_buffer.loss_weight

        if loss_scaler is None:
            loss.backward()
        else:
            # amp
            loss_scaler.scale(loss).backward()
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)

        if dyrep is not None:
            # record states of model in dyrep
            dyrep.record_metrics()
            
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s'.format(
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            lr=optimizer.param_groups[0]['lr'],
                            batch_time=batch_time_m,
                            data_time=data_time_m))
        if args.sched != "step":
            scheduler.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()

    return {'train_loss': loss_m.avg}



def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    batch_time_m = AverageMeter()
    start_time = time.time()


    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()
