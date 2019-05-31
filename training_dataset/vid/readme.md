# Preprocessing VID(Object detection from video)
Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)

### Download dataset (86GB)

````shell
wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Annotations/VID/train/a
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Annotations/VID/train/b
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Annotations/VID/train/c
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Annotations/VID/train/d
ln -sfb $PWD/ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/e

ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Data/VID/train/a
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Data/VID/train/b
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Data/VID/train/c
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Data/VID/train/d
ln -sfb $PWD/ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/e
````

### Crop & Generate data info (20 min)

````shell
python parse_vid.py

#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
