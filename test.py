import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse, time, cv2
from model.RAMENet import MALNet
from data_cod import test_dataset

# ======================
# Metrics functions
# ======================
def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta=0.3):
    pred_bin = (pred >= 0.5).astype(np.float32)
    tp = np.sum(pred_bin * gt)
    prec = tp / (np.sum(pred_bin) + 1e-8)
    rec = tp / (np.sum(gt) + 1e-8)
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-8)

def e_measure(pred, gt):
    pred_mean = pred.mean()
    gt_mean = gt.mean()
    align = 2 * (pred - pred_mean) * (gt - gt_mean) / ((pred - pred_mean)**2 + (gt - gt_mean)**2 + 1e-8)
    return np.mean(align)

def s_measure(pred, gt, alpha=0.5):
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = fg.mean() if fg.size > 0 else 0
    o_bg = bg.mean() if bg.size > 0 else 0
    return alpha * o_fg + (1 - alpha) * o_bg

# ======================
# Parse arguments
# ======================
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./Test/',help='test dataset path')
opt = parser.parse_args()

# ======================
# GPU setup
# ======================
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# ======================
# Load model
# ======================
model = MALNet()
model.load_state_dict(torch.load('/kaggle/input/ramanet/pytorch/default/1/EORSSD.pth'))
model.cuda()
model.eval()

# ======================
# Test datasets
# ======================
test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './pre_map/' + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    image_root = '/kaggle/input/eorssd/test-images/'
    gt_root = '/kaggle/input/eorssd/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    mae_sum = 0
    f_sum = 0
    e_sum = 0
    s_sum = 0
    cost_time = []

    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        start_time = time.perf_counter()
        outputs = model(image)          # model returns 8 outputs
        res = outputs[4]                # take the first sigmoid output
        cost_time.append(time.perf_counter() - start_time)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)

        # compute metrics
        mae_sum += mae(res, gt)
        f_sum += f_measure(res, gt)
        e_sum += e_measure(res, gt)
        s_sum += s_measure(res, gt)

    # Average metrics
    mae_avg = mae_sum / test_loader.size
    f_avg = f_sum / test_loader.size
    e_avg = e_sum / test_loader.size
    s_avg = s_sum / test_loader.size

    cost_time.pop(0)  # ignore first
    fps = test_loader.size / np.sum(cost_time)

    print(f"{dataset} - MAE: {mae_avg:.4f}, F-measure: {f_avg:.4f}, E-measure: {e_avg:.4f}, S-measure: {s_avg:.4f}")
    print(f"Mean running time: {np.mean(cost_time):.4f}s, FPS: {fps:.2f}")
