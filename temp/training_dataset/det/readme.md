# Preprocessing DET(Object detection)
Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)

### Download dataset (49GB)

````shell
wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz
tar -xzvf ./ILSVRC2015_DET.tar.gz

ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2013_train ILSVRC/Annotations/DET/train/a
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0000 ILSVRC/Annotations/DET/train/b
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0001 ILSVRC/Annotations/DET/train/c
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0002 ILSVRC/Annotations/DET/train/d
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0003 ILSVRC/Annotations/DET/train/e
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0004 ILSVRC/Annotations/DET/train/f
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0005 ILSVRC/Annotations/DET/train/g
ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0006 ILSVRC/Annotations/DET/train/h
ln -sfb $PWD/ILSVRC/Annotations/DET/val ILSVRC/Annotations/DET/train/i

ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2013_train ILSVRC/Data/DET/train/a
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0000 ILSVRC/Data/DET/train/b
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0001 ILSVRC/Data/DET/train/c
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0002 ILSVRC/Data/DET/train/d
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0003 ILSVRC/Data/DET/train/e
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0004 ILSVRC/Data/DET/train/f
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0005 ILSVRC/Data/DET/train/g
ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0006 ILSVRC/Data/DET/train/h
ln -sfb $PWD/ILSVRC/Data/DET/val ILSVRC/Data/DET/train/i
````

### Crop & Generate data info (20 min)

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````