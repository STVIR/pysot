# PySOT: Python Single Object Tracking

<div align="center">
  <img src="asset/demo/bag_demo.gif" width="800px" alt="Example outputs of SiamFC, SiamRPN, and SiamMask"/>
  <p>Example outputs of SiamFC, SiamRPN, and SiamMask.</p>
</div>

## Introduction

PySOT is a high-quality, high-performance codebase designed for visual tracking research. It facilitates the rapid implementation and evaluation of novel research ideas. PySOT includes implementations of the following state-of-the-art visual tracking algorithms:

- [SiamFC](https://arxiv.org/abs/1606.09549)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [SiamMask](https://arxiv.org/abs/1812.05050)

These algorithms are supported by powerful backbone network architectures:

- [ResNet (18, 34, 50)](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

For further details on these models and architectures, see the [References](#references) section.

## Supported Datasets

PySOT's evaluation toolkit supports the following datasets:

- [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf)
- [VOT16/18/19](http://votchallenge.net)
- [VOT18-LT](http://votchallenge.net/vot2018/index.html)
- [LaSOT](https://arxiv.org/pdf/1809.07845.pdf)
- [UAV123](https://arxiv.org/pdf/1804.00518.pdf)

## Roadmap

- [x] Simplify inference procedures via CLI.
- [ ] Streamline evaluation processes via CLI.
- [ ] Automate data downloads (model weights, datasets, videos).
- [ ] Simplify training setup via CLI.
- [ ] Release first official package.
- [ ] Expand documentation.

For detailed plans, refer to our [Roadmap Document](asset/roadmap/TODO.md).

## Quick Start

### Environment Setup

```bash
git clone https://github.com/MinLee0210/pysot.git
cd pysot
pip install -r requirements.txt
```

### Running Inference

```bash
python -m pysot --model_name="<model_name>" --video_name="<video_name>"
```

> **Note:** Video paths can be local or URL-based (YouTube links preferred).

## References

For comprehensive details on the technologies and methodologies used in PySOT, please consult the following publications:

- [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1812.05050) - IEEE CVPR, 2019
- [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/abs/1812.11703) - IEEE CVPR, 2019
- [Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048) - ECCV, 2018
- [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) - IEEE CVPR, 2018
- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) - ECCV Workshops, 2016

## Contributors

- [Minh, Le Duc](https://github.com/MinLee0210)
- [Fangyi Zhang](https://github.com/StrangerZhang)
- [Qiang Wang](http://www.robots.ox.ac.uk/~qwang/)
- [Bo Li](http://bo-li.info/)
- [Zhiyuan Chen](https://zyc.ai/)
- [Jinghao Zhou](https://shallowtoil.github.io/)

## License

PySOT is released under the [Apache 2.0 license](https://github.com/MinLee0210/pysot/blob/master/LICENSE).
