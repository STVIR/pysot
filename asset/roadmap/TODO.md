# 🛠️ TODO List for Single Object Detection Toolkit

## ✅ Dataset Setup

Download and verify the following datasets:
  - [ ] VOT2016
  - [ ] VOT2018
  - [ ] VOT2018-LT
  - [ ] OTB100 (OTB2015)
  - [ ] UAV123
  - [ ] NFS
  - [ ] LaSOT
  - [ ] TrackingNet (evaluation on server only)
  - [ ] GOT-10k (evaluation on server only)
- [ ] Add dataset JSONs for evaluation (from Google Drive/BaiduYun if needed)
- [ ] Add dataset loading scripts for custom formats

## ⚙️ Environment & Setup

- [ ] Create `INSTALL.md` with PyTorch and dependency instructions
- [ ] Add Python virtual environment setup (`venv`, `conda`, etc.)
- [ ] Add `setup.py` for easy module import
- [ ] Add export command for PYTHONPATH setup


## 📦 Model Integration

- [ ] Build `experiments/` folder structure
- [ ] Add support for backbone networks:
  - [ ] ResNet (18, 34, 50)
  - [ ] MobileNetV2
  - [ ] AlexNet
- [ ] Integrate base models:
  - [ ] SiamFC
  - [ ] SiamRPN
  - [ ] DaSiamRPN
  - [ ] SiamRPN++
  - [ ] SiamMask
- [ ] Document links to pretrained models (Model Zoo)

## 🧪 Testing & Evaluation

- [ ] Implement test pipeline (`tools/test.py`)
- [ ] Add support for evaluating:
  - [ ] VOT-style datasets
  - [ ] OTB
  - [ ] UAV123
  - [ ] LaSOT
- [ ] Implement evaluation pipeline (`tools/eval.py`)
- [ ] Add ability to batch evaluate trackers
- [ ] Log and visualize results in `results/` folder


## 🎮 Demo & Utility

- [ ] Build `demo.py` for webcam and video testing
- [ ] Allow video file as input (fallback for no webcam)
- [ ] Include sample videos in `demo/` folder

## 🏗️ Training Support

- [ ] Add `TRAIN.md` for training instructions
- [ ] Implement training scripts
- [ ] Provide training config templates (`config.yaml`)
- [ ] Add data augmentation and loss functions


## 🧰 Toolkit Extensions

- [ ] Integrate [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) or similar for benchmarking
- [ ] Add support for new/custom datasets
- [ ] Write script for result format conversion


## 🧹 Bugfixes & Troubleshooting

- [ ] Document common errors in README:
  - [ ] `ModuleNotFoundError: No module named 'pysot'`
  - [ ] `ImportError: cannot import name region`
- [ ] Create troubleshooting section with solutions

---

## 📚 Documentation

- [ ] Write main `README.md`
- [ ] Add detailed usage examples
- [ ] Link to academic references
- [ ] Add contributor and license sections