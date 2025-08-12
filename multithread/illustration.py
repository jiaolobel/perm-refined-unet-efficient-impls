import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# versions = ["original", "noif", "norm", "noeigen"] 

version = "cppthreadv4"
# n_thread = [2, 4, 8, 16, 32]
n_thread = [8, ] # only for v4

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image

for num in n_thread:
    in_path = "E:/Research/experiment_results/efficient_glob_perm_rfn_unet/iiki2025/{}/l8/0/{}/a=80.0, b=0.03125, r=3.0".format(version, num)  # Perm. RFN. UNet w/o 
    image_path = "E:/Research/experiment_data/l8/false/"
    out_path = "visual/jpg/{}/{}".format(version, num)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("Create out path")

    for f in os.listdir(in_path):
        if os.path.splitext(f)[-1] == ".npz":
            print("Load {}".format(f))

            mask_name = os.path.join(in_path, f)
            image_name = os.path.join(image_path, f.replace('rfn.npz', 'sr_bands.png'))
            out_name = os.path.join(out_path, f.replace('rfn.npz', 'sr_bands_masked.jpg'))

            image = np.array(Image.open(image_name), dtype=np.float32)
            mask = np.load(mask_name)["arr_0"]

            cloud, cloud_color = 3, [115, 223, 255]
            shadow, shadow_color = 2, [38, 115, 0]

            mask_image = apply_mask(image, mask, cloud, cloud_color)
            mask_image = apply_mask(mask_image, mask, shadow, shadow_color)

            mask_image = np.uint8(mask_image)

            plt.imsave(out_name, mask_image)
            print("Write to {}.".format(out_name))