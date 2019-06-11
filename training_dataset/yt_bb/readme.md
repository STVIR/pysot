# Preprocessing Youtube-bb(YouTube-BoundingBoxes Dataset)

### Download raw label

````shell
wget https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz
wget https://research.google.com/youtube-bb/yt_bb_detection_validation.csv.gz

gzip -d ./yt_bb_detection_train.csv.gz
gzip -d ./yt_bb_detection_validation.csv.gz
````

### Download raw image by `youtube-bb-utility`(spend long time, 400GB)

````shell
git clone https://github.com/mehdi-shiba/youtube-bb-utility.git
cd youtube-bb-utility
pip install -r requirements.txt
# python download_detection.py [VIDEO_DIR] [NUM_THREADS]
python download_detection.py ../ 12
cd ..
````

### Crop & Generate data info (1 DAY)

````shell
python par_crop.py
python gen_json.py
````
