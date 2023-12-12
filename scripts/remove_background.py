import os
import cv2
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from rembg import remove
from segment_anything import SamPredictor, sam_model_registry

def load_image(path):
    x = cv2.imread(path, 1)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def load_image2(path):
    x = imageio.imread(path)
    if x.shape[-1] == 4:
        x = x[..., :3]
    return x

def add_margin(pil_img, color, size):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def _predict_bbox_for_img(image):
    image_nobg = remove(image.convert('RGBA'), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

def create_masks_given_bbox(in_root, out_root, sam_ckpt_path):

    if not os.path.exists(out_root):
        os.makedirs(out_root, exist_ok=True)

    crop_size, img_size = 200, 256
    sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt_path)
    predictor = SamPredictor(sam)

    files = sorted(os.listdir(in_root))
    for idx, file in tqdm(enumerate(files), total=len(files)):
        path = os.path.join(in_root, file)

        img = load_image(path)
        bbox = _predict_bbox_for_img(img)
        predictor.set_image(img)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
        )

        mask = masks[0].astype(np.uint8) * 255
        mask = np.stack([mask]*3, axis = -1)

        mask_ = mask.astype(np.float32) / 255.0
        masked_img = img.astype(np.float32) * mask_ + 255.0 * (1.0 - mask_)
        masked_img = np.clip(masked_img, 0.0, 255.0).astype(np.uint8)
        pil_img = Image.fromarray(masked_img)

        alpha_np = np.asarray(mask)
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)

        # Creating RGB image
        ref_img_ = pil_img.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)

        out_img = add_margin(ref_img_, color=(255, 255, 255), size=img_size)
        out_img.save(os.path.join(out_root, file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir', type=str,
        required=True, help='Path to the directory with sparse-view images of the object.'
    )
    parser.add_argument(
        '--out_dir', type=str,
        required=True, help='Path to the directory where masked images of the object should be stored.'
    )
    parser.add_argument(
        '--sam_ckpt_path', type=str,
        required=True, help='Path to the SAM checkpoint file corresponding to the vit_l model.'
    )
    args = parser.parse_args()
    create_masks_given_bbox(args.in_dir, args.out_dir, args.sam_ckpt_path)
