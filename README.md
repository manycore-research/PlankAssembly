<div align="center">

# PlankAssembly: Robust 3D Reconstruction from Three Orthographic Views with Learnt Shape Programs

<h4>
  <a href='https://github.com/Huenao' target='_blank'>Wentao Hu</a>*
  ·
  <a href='https://bertjiazheng.github.io/' target='_blank'>Jia Zheng</a>*
  ·
  <a href='https://github.com/Elsa-zhang' target='_blank'>Zixin Zhang</a>*
  ·
  <a href='https://yuan-xiaojun.github.io/Yuan-Xiaojun/' target='_blank'>Xiaojun Yuan</a>
  ·
  <a href='https://sai.sysu.edu.cn/teacher/teacher01/1385356.htm' target='_blank'>Jian Yin</a>
  ·
  <a href='https://zihan-z.github.io/' target='_blank'>Zihan Zhou</a>
</h4>

<h4>
  IEEE/CVF Conference on Computer Vision (ICCV), 2023
</h4>

<h5>
  These authors contributed equally to this work.
</h5>

[![arXiv](http://img.shields.io/badge/arXiv-2308.05744-B31B1B.svg)](https://arxiv.org/abs/2308.05744)
[![Conference](https://img.shields.io/badge/ICCV-2023-4b44ce.svg)](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_PlankAssembly_Robust_3D_Reconstruction_from_Three_Orthographic_Views_with_Learnt_ICCV_2023_paper.html)

<img src="assets/teaser.gif">

</div>

> [!NOTE]
> This branch contains the implementation of PlankAssembly, which takes raster images as inputs. It is heavily built upon [Atlas](https://github.com/magicleap/Atlas). Specifically, we replace the Transformer encoder in PlankAssembly with a CNN-based feature extractor to construct a 3D feature volume from three-view images. Then, the Transformer decoder takes the flattened features as input and outputs the shape program.

## Setup

Our code has been tested with Python 3.8, PyTorch 1.10.0, CUDA 11.3 and PyTorch Lightning 1.7.6.

Please follow the instructions in the [main branch](https://github.com/manycore-research/PlankAssembly#installation) to setup your environment and download the dataset.

## Training

Follow the training steps of Atlas, download their provided pretrained resnet50 [weights](https://drive.google.com/file/d/15x8k-YOs_65N35CJafoJAPiftyX4Lx5w/view?usp=sharing) and unzip it. Then, use the following command to train a model from scratch:

```bash
python trainer.py fit --config configs/train.yaml
```

## Testing

Use the following command to test with our pretrained model ([weights](https://manycore-research-azure.kujiale.com/manycore-research/PlankAssembly/models/atlas-checkpoint_049-precision=0.808-recall=0.773-f1=0.787.ckpt)) or your own checkpoint:

```bash
python trainer.py test \
    --config configs/train.yaml \
    --ckpt_path path/to/checkpoint.ckpt \
    --trainer.devices 1
```

## Evaluation

To compute the evaluation metrics, please run the following command:

```bash
python evaluate.py --data_path path/to/data/dir --exp_path path/to/lightning_log/dir
```

## Visualization

To visualize the results, we build 3D mesh models from predictions:

```bash
python misc/build_pred_mesh.py --exp_path path/to/lightning_log/dir
```

Then, we use [HTML4Vision](https://github.com/mtli/HTML4Vision) to generate HTML files for mesh visualization (more details please refer to [here](https://github.com/mtli/HTML4Vision/#3d-models)):

```bash
python misc/build_html.py --exp_path path/to/lightning_log/dir
```

## LICENSE

PlankAssembly is licensed under the [AGPL-3.0 license](LICENSE). The code snippets in the [atlas](atlas) folder and the [third_party](third_party) are available under [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).