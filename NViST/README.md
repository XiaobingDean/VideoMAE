## Dataset

Download MVImgNet dataset from this [official repository](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet).
Highly recommend to use [Tip](https://docs.google.com/document/d/1krVb4B3rZw-0FaBBPS7c3SJKfqq5AVYTs2HN2LnlBPQ/edit#heading=h.2ukfzxh5c9pq) provided by the authors.

For the paper, they use the subset of MVImgNet - 1.14M frames, 38K scenes of 177 categories for training, and for testing, a total of 13,228 frames from 447 scenes and 177 categories are used. 

## Setting up an environment

This code is tested for Pytorch 2.4.0 with CUDA 11.8.

```sh
pip install -r requirements.txt
```
### Training NViST

We support multi-GPUs operation using accelerate. Both batch size (number of images for encoder) and batch pixel sizes (number of pixels we use for rendering) are for 40GB A100 GPUs. Increase the learning rate by $\sqrt{N}$ if you increase your batch size and batch pixel size by $N$.

```sh

accelerate launch --mixed_precision=fp16 scripts/train_nvist.py --config configs/mvimgnet_nvist_test.txt --batch_size 2 --batch_pixel_size 330000 --expname nvist_mvimgnet_2gpus --lr_encoder_init 0.00006 --lr_decoder_init 0.0003 --lr_renderer_init 0.0003

```
For 1 GPU

```sh

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision=fp16 scripts/train_nvist_test.py --config configs/mvimgnet_nvist.txt\
 --batch_size 11 --batch_pixel_size 165000 --expname nvist_mvimgnet_1gpu
```
