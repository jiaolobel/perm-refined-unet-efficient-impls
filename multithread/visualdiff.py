import numpy as np
import os

version = ["cppthreadv2", "cppthreadv3", "cppthreadv4"][1]
n_thread = [2, 4, 8, 16, 32]

masks = []

for num in n_thread:
    in_path = "E:/Research/experiment_results/efficient_glob_perm_rfn_unet/iiki2025/{}/l8/0/{}/a=80.0, b=0.03125, r=3.0".format(version, num)  # Perm. RFN. UNet w/o 

    for f in os.listdir(in_path):
        if os.path.splitext(f)[-1] == ".npz":
            print("Load {}".format(os.path.join(in_path, f)))

            mask_name = os.path.join(in_path, f)

            masks.append(np.load(mask_name)["arr_0"])

print(len(masks))

print(np.allclose(masks[0], masks[1]))
print(np.allclose(masks[1], masks[2]))
print(np.allclose(masks[2], masks[3]))
print(np.allclose(masks[3], masks[4]))

version = ["cppthreadv2", "cppthreadv3", "cppthreadv4"]
num = 8 # n_thread = [2, 4, 8, 16, 32][2]

masks = []

for ver in version:
    in_path = "E:/Research/experiment_results/efficient_glob_perm_rfn_unet/iiki2025/{}/l8/0/{}/a=80.0, b=0.03125, r=3.0".format(ver, num)  # Perm. RFN. UNet w/o 

    for f in os.listdir(in_path):
        if os.path.splitext(f)[-1] == ".npz":
            print("Load {}".format(os.path.join(in_path, f)))

            mask_name = os.path.join(in_path, f)

            masks.append(np.load(mask_name)["arr_0"])

print(len(masks))

print(np.allclose(masks[0], masks[1]))
print(np.allclose(masks[1], masks[2]))