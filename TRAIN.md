# PySOT Training Tutorial

This implements training of SiamRPN with backbone architectures, such as ResNet, AlexNet.
### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/) (New link for cropped data, [BaiduYun](https://pan.baidu.com/s/1xvgzU0pjQXXgVeJnK7vHSg), extract code: 6dd5. **NOTE: Data in old link is not correct. Please use cropped data in this new link.**)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)

## Download pretrained backbones
Download pretrained backbones from [Google Drive](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `pretrained_models` directory

## Training

To train a model (SiamRPN++), run `train.py` with the desired configs:

```bash
cd experiments/siamrpn_r50_l234_dwxcorr_8gpu
```

### Multi-processing Distributed Data Parallel Training

Refer to [Pytorch distributed training](https://pytorch.org/docs/stable/distributed.html) for detailed description.

#### Single node, multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

#### Multiple nodes:
Node 1: (IP: 192.168.1.1, and has a free port: 2333) master node
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=192.168.1.1 \  # adjust your ip here
    --master_port=2333 \
    ../../tools/train.py
```
Node 2:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=8 \
    --master_addr=192.168.1.1 \
    --master_port=2333 \
    ../../tools/train.py
```

## Testing
After training, you can test snapshots on VOT dataset.
For `AlexNet`, you need to test snapshots from 35 to 50 epoch. 
For `ResNet`, you need to test snapshots from 10 to 20 epoch.

```bash 
START=10
END=20
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} \ 
    python -u ../../tools/test.py \
        --snapshot {} \
	--config config.yaml \
	--dataset VOT2018 2>&1 | tee logs/test_dataset.log
```

## Evaluation
```
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 4 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```
