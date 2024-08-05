# A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging (ECCV 2024)
Miao Cao, Lishun Wang, Huan Wang and Xin Yuan

<hr />

> **Abstract:** Video Snapshot Compressive Imaging (SCI) aims to use a low-speed 2D camera to capture high-speed scene as snapshot compressed measurements, followed by a reconstruction algorithm to reconstruct the high-speed video frames. State-of-the-art (SOTA) deep learning-based algorithms have achieved impressive performance, yet with heavy computational workload. Network quantization is a promising way to reduce computational cost. However, a direct low-bit quantization will bring large performance drop. To address this challenge, in this paper, we propose a simple low-bit quantization framework (dubbed Q-SCI) for the end-to-end deep learning-based video SCI reconstruction methods which usually consist of a feature extraction, feature enhancement and video reconstruction module. Specifically, we first design a high-quality feature extraction module and a precise video reconstruction module to extract and propagate high-quality features in the low-bit quantized model. In addition, to alleviate the information distortion of the Transformer branch in the quantized feature enhancement module, we introduce a shift operation on the query and key distributions to further bridge the performance gap. Comprehensive experimental results manifest that our Q-SCI framework can achieve superior performance, e.g., 4-bit quantized EfficientSCI-S derived by our Q-SCI framework can theoretically accelerate the real-valued EfficientSCI-S by 7.8X with only 2.3% performance gap on the simulation testing datasets.
<hr />

## Network Architecture
![Illustration of Q-SCI](/figure/network.png)

## Installation
Please see the [Installation Manual](docs/install.md) for Q-SCI Installation. 

## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/quantized_sci/quantized_sci.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/quantized_sci/quantized_sci.py 
```

## Testing Q-SCI on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/quantized_sci/quantized_sci.py --weights=checkpoints/8bit.pth
python tools/test.py configs/quantized_sci/quantized_sci.py --weights=checkpoints/4bit.pth
python tools/test.py configs/quantized_sci/quantized_sci.py --weights=checkpoints/3bit.pth
python tools/test.py configs/quantized_sci/quantized_sci.py --weights=checkpoints/2bit.pth
```
And, change the bit_depth in *configs/quantized_sci/quantized_sci.py*

## PSNR-Computational cost Comparison
<img src="/figure/psnr.png" alt="PSNR-Params" width="450" height="322">

# Reconstruction results of the real datasets
<video width="640" height="480" controls>
  <source src="/figure/real_results.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Citation

```
@inproceedings{cao2024simple,
  title={A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging},
  author={Cao, Miao and Wang, Lishun and Wang, Huan and Yuan, Xin},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```


