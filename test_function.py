from __future__ import print_function, division
import os
import time
from utils import *
import wandb
from matplotlib import pyplot
import cv2
import numpy as np
from utils.metrics import calc_error


def test(test_model, img_freq, dataloader, dataset, maxdisp=192, gpu=True, logfilename='results'):
    epe_sum = 0
    test_model = test_model.eval()
    Loss_list = []
    EPE_list = []
    D1_list = []
    Threshold3_list = []
    inference_time_list = []

    os.makedirs('./predictions', exist_ok=True)
    os.makedirs('./predictions_errors', exist_ok=True)
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):

            start_time = time.time()
            disp_ests, losses, EPEs, D1s, Threshold3s = test_sample(test_model, sample, maxdisp, gpu)
            inference_time = time.time() - start_time

            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(dataloader),
                                                    inference_time))

            losses = tensor2numpy(losses)
            EPEs = tensor2numpy(EPEs)
            D1s = tensor2numpy(D1s)
            Threshold3s = tensor2numpy(Threshold3s)

            Loss_list.append(losses)
            EPE_list.append(EPEs)
            D1_list.append(D1s * 100)
            Threshold3_list.append(Threshold3s * 100)
            inference_time_list.append(inference_time)
            # psm_metrics = calc_error(disp_ests[-1], sample['disparity'], 0, 192)
            # epe_sum += psm_metrics["psm_epe"].item()
            file_name = sample["left_filename"][-1]
            name = file_name.split('/')
            file_name = os.path.join("", '_'.join(name[2:]))

            if len(Loss_list) % img_freq == 0:

                disp_est_tn = disp_ests[-1]
                disp_est_np = tensor2numpy(disp_est_tn)
                error_map = disp_error_image_func.apply(disp_est_tn, sample["disparity"])
                error_map = tensor2numpy(error_map.permute(0, 2, 3, 1))
                top_pad_np = tensor2numpy(sample["top_pad"])
                right_pad_np = tensor2numpy(sample["right_pad"])
                left_filenames = sample["left_filename"]

                for disp_est, top_pad, right_pad, fn, er_disp, EPE, D1, Threshold3 in zip(disp_est_np, top_pad_np,
                                                                                          right_pad_np, left_filenames,
                                                                                          error_map, EPEs, D1s, Threshold3s):
                    assert len(disp_est.shape) == 2
                    if dataset == 'kitti':
                        disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
                    else:
                        disp_est = np.array(disp_est, dtype=np.float32)
                    name = fn.split('/')
                    fn = os.path.join("predictions", '_'.join(name[2:]))
                    fnerror = os.path.join("predictions_errors", '_'.join(name[2:]))

                    print("saving to", fn, disp_est.shape)
                    disp_est_uint = np.round(disp_est)
                    pyplot.imsave(fn, disp_est_uint, cmap='jet')

                    disp_est_uint = cv2.imread(fn)
                    disp_est_uint = cv2.putText(disp_est_uint, 'EPE: {:.2f}, D1: {:.2f}'.format
                    (np.round(EPE, 2), np.round(D1 * 100, 2)),
                                                (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    cv2.imwrite(fn, disp_est_uint)

                    pyplot.imsave(fnerror, er_disp)

                    er_disp = cv2.imread(fnerror)
                    er_disp = cv2.putText(er_disp, 'EPE: {:.2f}, D1: {:.2f}'.format
                    (np.round(EPE, 2), np.round(D1 * 100, 2)),
                                          (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    cv2.imwrite(fnerror, er_disp)

                    gwc_metrics = {'test_loss': losses[-1],
                               'test_EPE': EPEs[-1],
                               'test_D1': D1s[-1] * 100,
                               'test_Threshold3': Threshold3s[-1] * 100,
                               'test_inference_time': inference_time,
                               'test_step': batch_idx,
                               fn: [wandb.Image(fn), wandb.Image(fnerror), wandb.Image(sample["left"])]
                               }

                    #combined_metrics = {**psm_metrics, **gwc_metrics}
                    wandb.log(gwc_metrics)
            else:
                gwc_metrics = {'test_loss': losses[-1],
                               'test_EPE': EPEs[-1],
                               'test_D1': D1s[-1] * 100,
                               'test_Threshold3': Threshold3s[-1] * 100,
                               'test_inference_time': inference_time,
                               'test_step': batch_idx,
                               }

                #combined_metrics = {**psm_metrics, **gwc_metrics}
                wandb.log(gwc_metrics)


        wandb.log({'avg_test_loss': np.mean(np.array(Loss_list)),
                   'avg_test_EPE': np.mean(np.array(EPE_list)),
                   'avg_test_D1': np.mean(np.array(D1_list)),
                   'avg_test_Threshold3': np.mean(np.array(Threshold3_list)),
                   'avg_test_inference_time': np.mean(np.array(inference_time_list))
                   })
        print(epe_sum)
        print('avg_test_Loss:', np.mean(np.array(Loss_list)))
        print('avg_test_EPE:', np.mean(np.array(EPE_list)))
        print('avg_test_D1:', np.mean(np.array(D1_list)))
        print('avg_test_Threshold3:', np.mean(np.array(Threshold3_list)))
        print('avg_test_inference_time', np.mean(np.array(inference_time_list)))


# test one sample
@make_nograd_func
def test_sample(test_model, sample, maxdisp, gpu):
    if gpu:
        test_model.cuda()
        test_model.eval()
        if type(sample) == list:
            disp_ests = test_model(sample[0].cuda(), sample[1].cuda())
            disp_gt = sample[2].cuda()
        else:
            disp_ests = test_model(sample['left'].cuda(), sample['right'].cuda())
            disp_gt = sample['disparity'].cuda()
    else:
        test_model = test_model.to('cpu')
        test_model.eval()
        disp_ests = test_model(sample['left'].to('cpu'), sample['right'].to('cpu'))
        disp_gt = sample['disparity']

    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    losses = [F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True) for disp_est in disp_ests]
    # losses.append(F.smooth_l1_loss(logits[mask], disp_gt[mask]))
    EPEs = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    D1s = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    Threshold3s = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    return disp_ests, losses, EPEs, D1s, Threshold3s