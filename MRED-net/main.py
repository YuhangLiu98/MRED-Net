import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICES'] = '1'        ## 1/2 ,multi GPU     2,3

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))


    train_data_loader = get_loader(mode='train',
                            load_mode=args.load_mode,
                            saved_path=args.train_data_path,
                            test_patient='L10086',
                            patch_n=args.patch_n,
                            patch_size=args.patch_size,
                            transform=args.transform,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)
    test_data_loader = get_loader(mode='test',
                             load_mode=args.load_mode,
                             saved_path=args.test_data_path,
                             test_patient=args.test_patient,
                             patch_n=None,
                             patch_size=None,
                             transform=args.transform,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    if args.mode == 'train':
        solver = Solver(args, train_data_loader, test_data_loader)
        solver.train()
    elif args.mode == 'test':
        solver = Solver(args, test_data_loader=test_data_loader)
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--train_data_path', type=str, default='/home/ac/data/lyh/npy_img_piglet_train_0.50/')
    parser.add_argument('--test_data_path', type=str, default='/home/ac/data/lyh/npy_img_piglet_test_0.50/')
    parser.add_argument('--test_patient', type=str, default='L1') 
    parser.add_argument('--save_path', type=str, default='./save_0.50/')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--val_step', type=int, default=100)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)

    parser.add_argument('--patch_n', type=int, default=10)         ## 10
    parser.add_argument('--patch_size', type=int, default=64)    ## 64
    parser.add_argument('--batch_size', type=int, default=25)   ## batch size has to be very small if size=512,16
    parser.add_argument('--num_epochs', type=int, default=16000)  ## 200 or 2000
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=8000)  ## original 3000 then 8000
    parser.add_argument('--save_iters', type=int, default=1000)  ## the iterats~epochs*10 useless for now
    parser.add_argument('--test_epochs', type=int, default=10200)

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str)  ##, default=[2,3]
    parser.add_argument('--num_workers', type=int, default=0)         #
    parser.add_argument('--multi_gpu', type=bool, default=False) ## 2/2 ,multi GPU

    args = parser.parse_args()
    main(args)
