import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser('Class Incremental Learning Training and Testing', add_help=False)
    #seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    #num_bases
    parser.add_argument('--num_bases', type=int, default=50, help='number of bases')
    #increment
    parser.add_argument('--increment', type=int, default=10, help='increment')
    #backbone
    parser.add_argument('--backbone', type=str, default='resnet32', help='backbone')
    #batch_size
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    #input_size
    parser.add_argument('--input_size', type=int, default=32, help='input size')
    #color_jitter float 0.0-1.0 default 0.4
    parser.add_argument('--color_jitter', type=float, default=0.4, help='color jitter')
    #aa AutoAugment Policy default rand-m9-mstd0.5-inc1 metavar='NAME'
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='AutoAugment Policy', metavar='NAME')
    #reprob
    parser.add_argument('--reprob', type=float, default=0.0, help='random erase prob')
    #remode
    parser.add_argument('--remode', type=str, default='pixel', help='random erase mode')
    #recount
    parser.add_argument('--recount', type=int, default=1, help='random erase count')
    #resplit action store true default false
    parser.add_argument('--resplit', action='store_true', help='resplit', default=False)
    #herding_method 
    parser.add_argument('--herding_method', type=str, default='barycenter', help='herding method')
    #memory_size
    parser.add_argument('--memory_size', type=int, default=2000, help='memory size')
    #fixed_memory
    parser.add_argument('--fixed_memory', action='store_true', help='fixed memory', default=False)
    #lr
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    #momentum
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    #weight_decay
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    #num_epochs
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    #smooth
    parser.add_argument('--smooth', type=float, default=0.0, help='smooth')
    #eval_every_epoch
    parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval every epoch')
    #dist_url
    parser.add_argument('--dist_url', default='env://', help='dist url')
    #data_set
    parser.add_argument('--data_set', type=str, default='cifar', help='data set')
    #data_path
    parser.add_argument('--data_path', type=str, default='data/data/data/cifar100', help='data path')
    #lambda_kd
    parser.add_argument('--lambda_kd', type=float, default=0.5, help='lambda kd')
    #dynamic_lambda_kd
    parser.add_argument('--dynamic_lambda_kd', action='store_true', help='dynamic lambda kd', default=False)

    return parser

