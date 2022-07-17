import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import timm
from continuum import rehearsal
from loss import SoftTarget
from model import CilModel

from opt import get_arg_parser
from utils import MetricLogger, build_dataset, init_distributed_mode, init_seed

@torch.no_grad()
def eval(model, val_loader):
    metric_logger = MetricLogger(delimiter="  ")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for images, target, task_ids in val_loader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        logits, _ = model(images)
        loss = criterion(logits, target)
        acc1, acc5 = timm.utils.accuracy(
            logits, target, topk=(1, min(5, logits.shape[1])))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print(' Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return metric_logger.acc1.global_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', parents=[get_arg_parser()])
    args = parser.parse_args()
    
    # init distributed mode
    init_distributed_mode(args)
    
    # init seed 
    init_seed(args)

    # class order
    args.class_order = [ 68, 56, 78, 8,
        23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

    scenario_train, args.num_classes = build_dataset(it_train=True, args=args)
    scenario_val, _ = build_dataset(it_train=False, args=args)

    model = CilModel(args)
    model = model.cuda()
    model_without_ddp = model

    torch.distributed.barrier()

    memory = rehearsal.RehearsalMemory(
        memory_size=args.memory_size,
        herding_method=args.herding_method,
        fixed_memory=args.fixed_memory
    )

    teacher_model = None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smooth)

    kd_criterion = SoftTarget(T=2)

    args.increment_per_task = [args.num_bases] + \
        [args.increment for _ in range(len(scenario_train)-1)]
    
    args.known_classes = 0
    acc1s = []

    

    for task_id, dataset_train in enumerate(scenario_train):
        args.task_id = task_id

        dataset_val = scenario_val[:task_id + 1]
        if task_id > 0:
            dataset_train.add_samples(*memory.get())
        
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        train_loader = DataLoader(
            dataset_train, # 
            batch_size=args.batch_size, # batch_size is set to 1 for teacher model
            sampler=train_sampler,  
            num_workers=args.workers, # num_workers is set to 0 for teacher model
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.workers,
        )

        model_without_ddp.prev_model_adaption(args.increment_per_task[task_id])

        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp, device_ids=[args.rank])
        
        optimizer = torch.optim.SGD(
            model_without_ddp.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = args.num_epochs
        )

        for epoch in range(args.num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            metric_logger = MetricLogger(delimiter="  ")
            for idx, (inputs, targets, task_ids) in enumerate(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                logits, _ = model(inputs)
                loss_ce = criterion(logits, targets)
                if teacher_model is not None:
                    t_logits, _ = teacher_model(inputs)
                    loss_kd = args.lambda_kd * \
                        kd_criterion(logits[:, :args.known_classes] , t_logits)
                else:
                    loss_kd = torch.tensor(0.0).cuda(non_blocking=True)
                loss = loss_ce + loss_kd
                acc1, acc5 = timm.utils.accuracy(
                    logits, targets, topk=(1, min(5,logits.shape[1])))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.distributed.barrier()
                metric_logger.update(ce=loss_ce)
                metric_logger.update(kd=loss_kd)
                metric_logger.update(acc1=acc1)
                metric_logger.update(loss=loss)

            metric_logger.synchronize_between_processes()
            lr_scheduler.step()

            print(
                f'train states: epoch: [{epoch}/{args.num_epochs}] {metric_logger}'
            )

            if (epoch + 1) % args.eval_every_epoch == 0:
                eval(model, val_loader)
            
        model_without_ddp.after_model_adaption(
            args.increment_per_task[task_id], args)
        
        acc1 = eval(model, val_loader)
        acc1s.append(acc1)

        print(f'task id = {task_id} @Acc1 = {acc1:.5f}, @Acc5 = {acc1s}')

        teacher_model = model_without_ddp.copy().freeze()
        unshuffle_train_loader = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=False)
        features = []
        for i, (inputs, labels, task_ids) in enumerate(unshuffle_train_loader):
            inputs = inputs.cuda(non_blocking=True)
            features.append(model_without_ddp.extract_vector(
                inputs).detach().cpu().numpy())
        features = np.concatenate(features, axis=0)
        memory.add(
            *dataset_train.get_raw_samples(), features
        )
        args.known_classes += args.increment_per_task[task_id]

        

        


