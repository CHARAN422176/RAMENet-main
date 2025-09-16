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
def mae_metric(pred, gt):
    return np.mean(np.abs(pred - gt))

def iou_metric(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

def f_measure(pred, gt, beta2=0.3):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

def s_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    alpha = 0.5
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = np.mean(fg) if fg.size > 0 else 0
    o_bg = np.mean(bg) if bg.size > 0 else 0
    object_score = alpha * o_fg + (1 - alpha) * (1 - o_bg)
    h, w = gt.shape
    y, x = h // 2, w // 2
    gt_quads = [gt[:y, :x], gt[:y, x:], gt[y:, :x], gt[y:, x:]]
    pr_quads = [pred[:y, :x], pred[:y, x:], pred[y:, :x], pred[y:, x:]]
    region_score = 0
    for gq, pq in zip(gt_quads, pr_quads):
        region_score += np.mean(1 - np.abs(pq - gq))
    region_score /= 4.0
    return 0.5 * (object_score + region_score)

def e_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    fm = np.mean(pred)
    gt_mean = np.mean(gt)
    align_matrix = 2 * (pred - fm) * (gt - gt_mean) / ((pred - fm) ** 2 + (gt - gt_mean) ** 2 + 1e-8)
    return np.mean((align_matrix + 1) ** 2 / 4)

# ======================
# Parse arguments
# ======================
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./Test/', help='test dataset path')
opt = parser.parse_args()

# ======================
# GPU setup
# ======================
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print(f'USE GPU {opt.gpu_id}')

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
        gt /= (gt.max() + 1e-8)  # normalize GT

        image = image.cuda()
        start_time = time.perf_counter()
        outputs = model(image)     # 8 outputs
        res = outputs[4]           # main sigmoid output
        cost_time.append(time.perf_counter() - start_time)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, (res * 255).astype(np.uint8))

        # Metrics
        pred_bin = (res >= 0.5).astype(np.float32)
        gt_bin = (gt >= 0.5).astype(np.float32)

        mae_sum += mae_metric(res, gt)
        f_sum += f_measure(pred_bin, gt_bin)
        e_sum += e_measure(res, gt)
        s_sum += s_measure(pred_bin, gt_bin)

    # Average metrics
    mae_avg = mae_sum / test_loader.size
    f_avg = f_sum / test_loader.size
    e_avg = e_sum / test_loader.size
    s_avg = s_sum / test_loader.size

    cost_time.pop(0)  # remove first iteration
    fps = test_loader.size / np.sum(cost_time)

    print(f"{dataset} - MAE: {mae_avg:.4f}, F-measure: {f_avg:.4f}, E-measure: {e_avg:.4f}, S-measure: {s_avg:.4f}")
    print(f"Mean running time: {np.mean(cost_time):.4f}s, FPS: {fps:.2f}")
