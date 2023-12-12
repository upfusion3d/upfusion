# Extracting Masks

## Automatic Foreground Extraction

We build upon some heuristics used in [One-2-3-45](https://github.com/One-2-3-45/One-2-3-45) and [SyncDreamer](https://liuyuan-pal.github.io/SyncDreamer/) to create an automatic foreground extraction helper script. This script first uses **rembg** to estimate an approximate bounding box where the object might belong. Then, the bounding box is used by Segment Anything Model (**SAM**) to extract a refined mask. To use the script to create masked input images, please follow the below steps:

1. Make sure the environment is setup according to the instructions in `README.md`. Then, install the following additional requirements:
	- Install rembg using `rembg>=2.0.50`
	- Install requirements needed for SAM following the instructions in their [repository](https://github.com/facebookresearch/segment-anything).
	- Download weights for the `vit_l` SAM model type following the instructions in their [repository](https://github.com/facebookresearch/segment-anything).

2. Place all the input images in some directory. Then, you can run the helper script using the below command (please replace `/path/to` with the appropriate paths).
```bash
python -m scripts.remove_background --in_dir /path/to/input_dir --out_dir /path/to/output_dir --sam_ckpt_path /path/to/sam/weights.pth
```

The output directory should contain masked images with white background of size `(256, 256)`.

## Limitations and Recommendations

The automatic foreground extraction script may not work incredibly well for difficult objects. In that case, we recommend users to provide additional prompts to SAM to enhance the quality of the extracted mask. Users can refer to this [link](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb) for a brief guide on how to guide SAM to provide better masks by providing additional point annotations.
