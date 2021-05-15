import numpy as np
from PIL import Image
import pdb
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

gt_gif = Image.open("/Users/gparmar/Desktop/gt_edit3.gif")
baseline_gif = Image.open("/Users/gparmar/Desktop/layer0_neuron18_mul1000.gif")
ours_gif =  Image.open("/Users/gparmar/Desktop/edits_3.gif")
# pdb.set_trace()

# POSE #1
gt_gif.seek(1), baseline_gif.seek(0), ours_gif.seek(1)
pil_gt = np.asarray(gt_gif.convert("RGB"))
pil_base = np.asarray(baseline_gif.convert("RGB"))
pil_ours = np.asarray(ours_gif.convert("RGB"))
Image.fromarray(pil_gt).save("tmp/_gt_1.png"), Image.fromarray(pil_base).save("tmp/baseline_1.png"), Image.fromarray(pil_ours).save("tmp/ours_1.png")
score = peak_signal_noise_ratio(pil_gt, pil_base)
print(f"POSE#1 PSNR w.r.t. baseline = {score:.3f}")
score = mean_squared_error(pil_gt, pil_base)
print(f"POSE#1 MSE w.r.t. baseline = {score:.3f}")

score = peak_signal_noise_ratio(pil_gt, pil_ours)
print(f"POSE#1 PSNR w.r.t. ours = {score:.3f}")
score = mean_squared_error(pil_gt, pil_ours)
print(f"POSE#1 MSE w.r.t. ours = {score:.3f}")

# POSE #2
gt_gif.seek(8), baseline_gif.seek(7), ours_gif.seek(8)
pil_gt = np.asarray(gt_gif.convert("RGB"))
pil_base = np.asarray(baseline_gif.convert("RGB"))
pil_ours = np.asarray(ours_gif.convert("RGB"))
Image.fromarray(pil_gt).save("tmp/_gt_2.png"), Image.fromarray(pil_base).save("tmp/baseline_2.png"), Image.fromarray(pil_ours).save("tmp/ours_2.png")
print(f"POSE#2 PSNR w.r.t. baseline = {score:.3f}")
score = mean_squared_error(pil_gt, pil_base)
print(f"POSE#2 MSE w.r.t. baseline = {score:.3f}")
score = peak_signal_noise_ratio(pil_gt, pil_ours)
print(f"POSE#2 PSNR w.r.t. ours = {score:.3f}")
score = mean_squared_error(pil_gt, pil_ours)
print(f"POSE#2 MSE w.r.t. ours = {score:.3f}")