#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import json
from os.path import join, exists
import pandas as pd

# The data sets to be downloaded
d_sets = ['yt_bb_detection_validation', 'yt_bb_detection_train']

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

instanc_size = 511
crop_path = './crop{:d}'.format(instanc_size)


def parse_and_sched(dl_dir='.'):
    # For each of the two datasets
    js = {}
    for d_set in d_sets:

        # Make the directory for this dataset
        d_set_dir = dl_dir+'/'+d_set+'/'

        # Parse csv data using pandas
        print (d_set+': Parsing annotations into clip data...')
        df = pd.DataFrame.from_csv(d_set+'.csv', header=None, index_col=False)
        df.columns = col_names

        # Get list of unique video files
        vids = df['youtube_id'].unique()

        for vid in vids:
            data = df[df['youtube_id']==vid]
            for index, row in data.iterrows():
                youtube_id, timestamp_ms, class_id, class_name, \
                object_id, object_presence, x1, x2, y1, y2 = row

                if object_presence == 'absent':
                    continue

                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or y2 < y1 or x2 < x1:
                    continue

                bbox = [x1, y1, x2, y2]
                frame = '%06d' % (int(timestamp_ms) / 1000)
                obj = '%02d' % (int(object_id))
                video = join(d_set_dir + str(class_id), youtube_id)

                if not exists(join(crop_path, video, '{}.{}.x.jpg'.format(frame, obj))):
                    continue

                if video not in js:
                    js[video] = {}
                if obj not in js[video]:
                    js[video][obj] = {}
                js[video][obj][frame] = bbox

        if 'yt_bb_detection_train' == d_set:
            json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
        else:
            json.dump(js, open('val.json', 'w'), indent=4, sort_keys=True)
        js = {}
        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
