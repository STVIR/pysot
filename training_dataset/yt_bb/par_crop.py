#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import os
from os.path import join
import sys
import cv2
import pandas as pd
import numpy as np

# The data sets to be downloaded
d_sets = ['yt_bb_detection_validation', 'yt_bb_detection_train']

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'


# Print iterations progress (thanks StackOverflow)
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


instanc_size = 511
crop_path = './crop{:d}'.format(instanc_size)
check_call(['mkdir', '-p', crop_path])


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


# Download and cut a clip to size
def dl_and_cut(vid, data, d_set_dir):
    for index, row in data.iterrows():
        youtube_id, timestamp_ms, class_id, class_name,\
        object_id, object_presence, xmin, xmax, ymin, ymax = row

        if object_presence == 'absent':
            continue

        class_dir = d_set_dir + str(class_id)
        frame_path = class_dir + '/' + youtube_id + '_' + str(timestamp_ms) + \
                     '_' + str(class_id) + '_' + str(object_id) + '.jpg'
        # Verify that the video has been downloaded. Skip otherwise
        if not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)
        avg_chans = np.mean(image, axis=(0, 1))
        # Uncomment lines below to print bounding boxes on downloaded images
        h, w = image.shape[:2]
        x1 = xmin*w
        x2 = xmax*w
        y1 = ymin*h
        y2 = ymax*h
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or y2 < y1 or x2 < x1:
            continue

        # Make the class directory if it doesn't exist yet
        crop_class_dir = join(crop_path, d_set_dir+str(class_id), youtube_id)
        check_call(['mkdir', '-p', crop_class_dir])

        # Save the extracted image
        bbox = [x1, y1, x2, y2]
        z, x = crop_like_SiamFC(image, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(crop_class_dir, '{:06d}.{:02d}.z.jpg'.format(int(timestamp_ms)/1000, int(object_id))), z)
        cv2.imwrite(join(crop_class_dir, '{:06d}.{:02d}.x.jpg'.format(int(timestamp_ms)/1000, int(object_id))), x)
    return True

       
# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='.', num_threads=24):
    """Crop the entire youtube-bb data set into `crop_path`.
    """
    # For each of the two datasets
    for d_set in d_sets:

        # Make the directory for this dataset
        d_set_dir = dl_dir+'/'+d_set+'/'

        # Download & extract the annotation list
        # print (d_set+': Downloading annotations...')
        # check_call(['wget', web_host+d_set+'.csv.gz'])
        # print (d_set+': Unzipping annotations...')
        # check_call(['gzip', '-d', '-f', d_set+'.csv.gz'])

        # Parse csv data using pandas
        print (d_set+': Parsing annotations into clip data...')
        df = pd.DataFrame.from_csv(d_set+'.csv', header=None, index_col=False)
        df.columns = col_names

        # Get list of unique video files
        vids = df['youtube_id'].unique()

        # Download and cut in parallel threads giving
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut,vid,df[df['youtube_id']==vid],d_set_dir) for vid in vids]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, len(vids),
                            prefix = d_set,
                            suffix = 'Done',
                            barLength = 40)

        print(d_set+': All videos Crop Done')


if __name__ == '__main__':
    parse_and_sched()
