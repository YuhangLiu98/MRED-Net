import os
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from thop import profile
from thop import clever_format
from prep import printProgressBar
# from models.vmunet.vmunet import VMUNet
from models.vmunet.MREDnet import VSSM

from loss import Disentangle_UNet, init_net, define_F, PatchNCELoss

from measure import compute_measure,compute_SSIM
import logging
import datetime
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
# def split_arr(arr,patch_size,stride=64):    ## 512*512 to 32*32
#     pad = (32, 32, 32, 32) # pad by (0, 1), (2, 1), and (3, 3)
#     arr = nn.functional.pad(arr, pad, "constant", 0)
#     _,_,h,w = arr.shape
#     num = h//stride - 1
#     arrs = torch.zeros(num*num,1,patch_size,patch_size)
#
#     for i in range(num):
#         for j in range(num):
#             arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
#     return arrs
#
# def agg_arr(arrs, size, stride=64):  ## from 32*32 to size 512*512
#     arr = torch.zeros(size, size)
#     n,_,h,w = arrs.shape
#     num = size//stride
#     for i in range(num):
#         for j in range(num):
#             arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,32:32+64,32:32+64]
#   #return arr
#     return arr.unsqueeze(0).unsqueeze(1)


def split_arr(arr, patch_size, stride=32):  ## 512*512 to 32*32
    pad = (16, 16, 16, 16)  # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _, _, h, w = arr.shape
    num = h // stride - 1
    arrs = torch.zeros(num * num, 1, patch_size, patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i * num + j, 0] = arr[0, 0, i * stride:i * stride + patch_size, j * stride:j * stride + patch_size]
    return arrs


def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n, _, h, w = arrs.shape
    num = size // stride
    for i in range(num):
        for j in range(num):
            arr[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride] = arrs[i * num + j, :, 16:48, 16:48]
    # return arr
    return arr.unsqueeze(0).unsqueeze(1)


class Solver(object):
    def __init__(self, args, train_data_loader=None, test_data_loader=None):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.val_step = args.val_step
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_epochs = args.test_epochs
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        current_time = datetime.datetime.now()
        log_filename = os.path.join(self.save_path, current_time.strftime("%Y-%m-%d_%H-%M-%S") + '_training.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

        print('#----------Prepareing Model----------#')
        # self.VMUNet = VMUNet(
        #     num_classes=1,
        #     input_channels=1,
        #     depths=[1],
        #     depths_decoder=[1],
        #     dims=[64],
        #     dims_decoder=[64],
        #     drop_path_rate=0.2,
        #     load_ckpt_path=None,
        # )
        self.VMUNet = VSSM(in_chans=1,
                           num_classes=1,
                           depths=[1, 1],
                           depths_decoder=[1, 1],
                           dims=[32, 32],
                           dims_decoder=[32, 32],
                           drop_path_rate=0.2
                           )
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.VMUNet = nn.DataParallel(self.VMUNet)  ## data parallel  ,device_ids=[2,3]
        self.VMUNet.to(self.device)

        self.netPredictor = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        ).to(self.device)

        netOnline = Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
        self.netOnline = init_net(netOnline, gpu_ids=[0], initialize_weights=True)
        netTarget = Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
        self.netTarget = init_net(netTarget, gpu_ids=[0], initialize_weights=True)

        self.netProjection_online = define_F(1, 'mlp_sample', 'batch', False,
                                                      'xavier', 0.02, True, [0],
                                                      initialize_weights=True).to(self.device)
        self.netProjection_target = define_F(1, 'mlp_sample', 'batch', False,
                                                      'xavier', 0.02, True, [0],
                                                      initialize_weights=True).to(self.device)
        self.initializes_target_network()
        self.criterionNCE = []
        nce_layers = [1, 4]
        for nce_layer in nce_layers:
            self.criterionNCE.append(PatchNCELoss().to(self.device))


        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.VMUNet.parameters(), self.lr)

        self.optimizer_R = torch.optim.SGD(
            list(self.netOnline.parameters()) + list(self.netProjection_online.parameters()) + list(
                self.netPredictor.parameters()), lr=self.lr)
        self.SPloss= torch.nn.MSELoss()

    def save_model(self, epoch_):
        f = os.path.join(self.save_path, 'Mamba_{}epoch.ckpt'.format(epoch_))
        torch.save(self.VMUNet.state_dict(), f)

    def load_model(self, epoch_):
        # device = torch.device('cpu')
        device = torch.device('cuda')

        f = os.path.join(self.save_path, 'Mamba_{}epoch.ckpt'.format(epoch_))
        self.VMUNet.load_state_dict(torch.load(f, map_location=device))

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        for param_group_R in self.optimizer_R.param_groups:
            param_group_R['lr'] = lr
    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def train(self):
        NumOfParam = count_parameters(self.VMUNet)
        print('trainable parameter:', NumOfParam)

        train_losses = []
        total_iters = 0
        start_time = time.time()
        loss_all = []
        self.load_model(5700)

        for epoch in range(1, self.num_epochs):
            self.VMUNet.train(True)

            for iter_, (x, y) in enumerate(self.train_data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)  ## expand one dimension given the dimension 0  4->[1,4]
                y = y.unsqueeze(0).float().to(self.device)  ## copy data to device

                if self.patch_size:  # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)  ## similar to reshape
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.VMUNet(x)

                self.set_requires_grad(self.netOnline, True)
                self.set_requires_grad(self.netProjection_online, True)
                self.set_requires_grad(self.netPredictor, True)
                self.optimizer_R.zero_grad()
                self.loss_D = self.compute_D_loss(pred, y)
                self.loss_D.backward()
                self.optimizer_R.step()
                self._update_target_network_parameters()

                # update G
                self.set_requires_grad(self.netOnline, False)
                self.set_requires_grad(self.netProjection_online, False)
                self.set_requires_grad(self.netPredictor, False)

                self.optimizer.zero_grad()
                self.loss_G = self.compute_G_loss(pred, y)
                self.loss_G.backward()
                self.optimizer.step()

                # print(pred.shape)
                # loss = self.criterion(pred, y) * 100 + 1e-4  ## to prevent 0

                # self.VMUNet.zero_grad()
                # self.optimizer.zero_grad()
                #
                # loss.backward()
                # self.optimizer.step()
                train_losses.append(self.loss_G.item())

                loss_all.append(self.loss_G.item())
                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters,
                                                                                                        epoch,
                                                                                                        self.num_epochs,
                                                                                                        iter_ + 1,
                                                                                                        len(self.train_data_loader),
                                                                                                        self.loss_G.item(),
                                                                                                        time.time() - start_time))
                # learning rate decay
                # print(total_iters)
                # if total_iters % self.decay_iters == 0:
            if epoch % 10 == 0:
                self.lr_decay()
                # save model
                # if total_iters % 1000 == 0:   # 20000

            # network validation
            if epoch % self.val_step == 0:
                # save model
                print("save model: ", total_iters)
                self.save_model(epoch)
                np.save(os.path.join(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))
                # validation
                self.val(epoch)
        # self.save_model(epoch)
        np.save(os.path.join(self.save_path, 'loss_{}_epoch.npy'.format(epoch)), np.array(train_losses))
        print("total_iters:", total_iters)
        ## save loss figure
        plt.plot(np.array(loss_all), 'r')  ## print out the loss curve
        # plt.show()
        plt.savefig(self.save_path + '/loss.png')

    def val(self, epoch):
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        self.VMUNet.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                # arrs = split_arr(x, 128).to(self.device)  ## split to image patches for test into 4 patches
                # arrs = self.VMUNet(arrs)

                arrs = split_arr(x, 64).to(self.device)  ## split to image patches for test into 4 patches

                arrs[0:64] = self.VMUNet(arrs[0:64])
                arrs[64:2 * 64] = self.VMUNet(arrs[64:2 * 64])
                arrs[2 * 64:3 * 64] = self.VMUNet(arrs[2 * 64:3 * 64])
                arrs[3 * 64:4 * 64] = self.VMUNet(arrs[3 * 64:4 * 64])

                pred = agg_arr(arrs, 512).to(self.device)

                # pred = x - pred# denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                printProgressBar(i, len(self.test_data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            PSNR_AVG = pred_psnr_avg / len(self.test_data_loader)
            SSIM_AVG = pred_ssim_avg / len(self.test_data_loader)
            RMSE_AVG = pred_rmse_avg / len(self.test_data_loader)
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(PSNR_AVG, SSIM_AVG,
                                                                                                    RMSE_AVG))
            logging.info('Epoch %d: PSNR=%.4f, SSIM=%.4f, RMSE=%.4f', epoch, PSNR_AVG, SSIM_AVG, RMSE_AVG)

    def test(self):
        del self.VMUNet
        print('#----------Prepareing Model----------#')
        # self.VMUNet = VMUNet(
        #     num_classes=1,
        #     input_channels=1,
        #     depths=[1],
        #     depths_decoder=[1],
        #     dims=[64],
        #     dims_decoder=[64],
        #     drop_path_rate=0.2,
        #     load_ckpt_path=None,
        # )
        self.VMUNet = VSSM(in_chans=1,
                           num_classes=1,
                           depths=[1, 1],
                           depths_decoder=[1, 1],
                           dims=[32, 32],
                           dims_decoder=[32, 32],
                           drop_path_rate=0.2
                           )
        total_params = sum(p.numel() for p in self.VMUNet.parameters())
        model_parameters = filter(lambda p: p.requires_grad, self.VMUNet.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total parameters in the model: {total_params}")
        print(f"Trainable parameters in the model: {trainable_params}")

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            # print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.VMUNet = nn.DataParallel(self.VMUNet)  ## data parallel
        self.VMUNet.to(self.device)
        self.load_model(self.test_epochs)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        if self.result_fig == True:
            if not os.path.exists(self.save_path + '/C/'):
                os.makedirs(self.save_path + '/C/')
            else:
                print(self.save_path + '/C/' + 'exists!!!')
            if not os.path.exists(self.save_path + '/GT/'):
                os.makedirs(self.save_path + '/GT/')
            else:
                print(self.save_path + '/GT/' + 'exists!!!')
            if not os.path.exists(self.save_path + '/I/'):
                os.makedirs(self.save_path + '/I/')
            else:
                print(self.save_path + '/I/' + 'exists!!!')

        time_record = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                startTime = time.time()
                arrs = split_arr(x, 64).to(self.device)  ## split to image patches for test into 4 patches
                arrs[0:64] = self.VMUNet(arrs[0:64])
                arrs[64:2 * 64] = self.VMUNet(arrs[64:2 * 64])
                arrs[2 * 64:3 * 64] = self.VMUNet(arrs[2 * 64:3 * 64])
                arrs[3 * 64:4 * 64] = self.VMUNet(arrs[3 * 64:4 * 64])

                # arrs = split_arr(x, 128).to(self.device)  ## split to image patches for test into 4 patches
                # arrs = self.VMUNet(arrs)
                pred = agg_arr(arrs, 512).to(self.device)
                endTime = time.time()
                if i != 0:
                    time_record.append(endTime - startTime)

                # pred = x - pred# denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)
                    size = [1, 1, shape_ * shape_]
                    x = (x - torch.min(x.view(size), -1)[0].unsqueeze(-1)) / (
                            torch.max(x.view(size), -1)[0].unsqueeze(-1) -
                            torch.min(x.view(size), -1)[0].unsqueeze(-1))
                    pred = (pred - torch.min(pred.view(size), -1)[0].unsqueeze(-1)) / (
                            torch.max(pred.view(size), -1)[0].unsqueeze(-1) -
                            torch.min(pred.view(size), -1)[0].unsqueeze(-1))
                    y = (y - torch.min(y.view(size), -1)[0].unsqueeze(-1)) / (
                            torch.max(y.view(size), -1)[0].unsqueeze(-1) -
                            torch.min(y.view(size), -1)[0].unsqueeze(-1))

                    save_image(torch.clamp(pred, min=0, max=1).detach().cpu(),
                               os.path.join(self.save_path, 'C', '{}.png'.format(i)))
                    save_image(torch.clamp(y, min=0, max=1).detach().cpu(),
                               os.path.join(self.save_path, 'GT', '{}.png'.format(i)))
                    save_image(torch.clamp(x, min=0, max=1).detach().cpu(),
                               os.path.join(self.save_path, 'I', '{}.png'.format(i)))

                printProgressBar(i, len(self.test_data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print(f"每张图片平均用时：{sum(time_record) / len(time_record)}")

            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader),
                ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader),
                pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))
            input = torch.randn(256, 1, 64, 64).cuda()
            macs, params = profile(self.VMUNet.cuda(), (input,))
            macs, params = clever_format([macs, params], "%.3f")
            print(macs)
            print(params)

    @torch.no_grad()
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.netOnline.parameters(), self.netTarget.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.netProjection_online.parameters(), self.netProjection_target.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def compute_D_loss(self, pred, target):

        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D = self.MAC_Net(target, pred.detach())
        return self.loss_D

    def compute_G_loss(self, pred, target):
        """Calculate GAN and NCE loss for the generator"""
        #ipdb.set_trace()

        self.loss_NCE = self.MAC_Net(target, pred)
        # data_range = self.trunc_max - self.trunc_min

        self.loss_S = 10*self.SPloss(pred, target)+(1-compute_SSIM(pred, target, 1))
        self.loss_G = 0.1*self.loss_NCE + 10*self.loss_S

        return self.loss_G


    def MAC_Net(self, target, pred):
        nce_layers = [1, 4]
        num_patches = [32, 512]
        n_layers = len(nce_layers)

        patch_size = target.shape[-1]

        pixweght = target

        with torch.no_grad():
            feat_k_1 = self.netTarget(target, nce_layers, encode_only=True)
            feat_k_pool_1, sample_ids, sample_local_ids, sample_top_idxs = self.netProjection_target(patch_size,
                                                                                                     feat_k_1,
                                                                                                     num_patches,
                                                                                                     None, None, None,
                                                                                                     pixweght=None)

        feat_q_1 = self.netOnline(pred, nce_layers, encode_only=True)
        feat_q_pool_1, _, _, _ = self.netProjection_online(patch_size, feat_q_1, num_patches, sample_ids,
                                                           sample_local_ids, sample_top_idxs,
                                                           pixweght=pixweght)  # online

        total_nce_loss = 0.0
        for i, (f_q_1, f_k_1, crit) in enumerate(zip(feat_q_pool_1, feat_k_pool_1, self.criterionNCE)):
            if i == 0:
                loss = self.regression_loss(self.netPredictor(f_q_1), f_k_1.detach())
            if i == 1:
                loss = crit(f_q_1, f_k_1.detach())

            weight = torch.tensor(1.0)

            total_nce_loss += weight * loss.mean()

        return total_nce_loss

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.netOnline.parameters(), self.netTarget.parameters()):
            param_k.data = param_k.data * 0.99 + param_q.data * (1. - 0.99)

        for param_q, param_k in zip(self.netProjection_online.parameters(), self.netProjection_target.parameters()):
            param_k.data = param_k.data * 0.99 + param_q.data * (1. - 0.99)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad