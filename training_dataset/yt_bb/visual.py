import glob
import pandas as pd
import numpy as np
import cv2

visual = True

col_names = ['youtube_id', 'timestamp_ms', 'class_id', 'class_name',
             'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']

df = pd.DataFrame.from_csv('yt_bb_detection_validation.csv', header=None, index_col=False)
df.columns = col_names
frame_num = len(df['youtube_id'])

img_path = glob.glob('/mnt/qwang/youtubebb/frames/val*/*/*.jpg')
d = {key.split('/')[-1]: value for (value, key) in enumerate(img_path)}

for n in range(frame_num):
    if df['object_presence'][n]:
        frame_name = df['youtube_id'][n] + '_' + str(df['timestamp_ms'][n]) + '_' + \
                     str(df['class_id'][n]) + '_' + str(df['object_id'][n]) + '.jpg'
        bbox = np.array([df['xmin'][n],df['ymin'][n],df['xmax'][n],df['ymax'][n]])
        if frame_name in d.keys():
            frame_path = img_path[d[frame_name]]
            if visual:
                im = cv2.imread(frame_path)
                h, w, _ = im.shape
                pt1 = (int(bbox[0]*w), int(bbox[1]*h))
                pt2 = (int(bbox[2]*w), int(bbox[3]*h))
                cv2.rectangle(im, pt1, pt2, (0, 255, 0), 2)
                cv2.imshow('img', im)
                cv2.waitKey(100)
        else:
            print('no image: {}'.format(frame_name))
            pass
    else:
        pass

print('done')

