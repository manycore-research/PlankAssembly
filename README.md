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

<img width=500 src="assets/teaser.png">

</div>

> [!NOTE]
> This branch contains the implementation of comparison method, [PolyGen](https://arxiv.org/abs/2002.10880). Similar to our method, PolyGen adopts a Transformer-based architecture and proceeds by generating a set of 3D vertices, which are then connected to form 3D faces.

## Setup

Our code has been tested with Python 3.8, PyTorch 1.10.0, CUDA 11.3 and PyTorch Lightning 1.7.6.

Please follow the instructions in the [main branch](https://github.com/manycore-research/PlankAssembly#installation) to set up your environment and download the dataset.

## Data Processing

To train PolyGen, we first parse our shape program into vertex sequence and face sequence:

```bash
python dataset/preprocess.py
```

## Training

Use the following command to train vertex model and face model from scratch:

```bash
# train vertex model
python vertex_trainer.py fit --config configs/vertex.yaml
# train face model
python face_trainer.py fit --config configs/face.yaml
```

## Testing

Use the following command to **test** with our pretrained model ([vertex model weight](https://manycore-research-azure.kujiale.com/manycore-research/PlankAssembly/models/vertex-checkpoint.ckpt), [face model weight](https://manycore-research-azure.kujiale.com/manycore-research/PlankAssembly/models/face-checkpoint.ckpt)) or your own checkpoint:

```bash
# please first modify the checkpoint path in configs/test.yaml
python tester.py test --config configs/test.yaml
```

## Visualization

To visualize the results, we build 3D mesh models from predictions:

```bash
python postprocess.py --exp_path path/to/lightning_log/dir
```

## LICENSE

PlankAssembly is licensed under the [AGPL-3.0 license](LICENSE). The code snippets in the [third_party](third_party) are available under [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
