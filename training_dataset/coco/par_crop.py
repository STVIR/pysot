from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
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
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


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


def crop_img(img, anns, set_crop_base_path, set_img_base_path, instanc_size=511):
    frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
    avg_chans = np.mean(im, axis=(0, 1))
    for trackid, ann in enumerate(anns):
        rect = ann['bbox']
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if rect[2] <= 0 or rect[3] <=0:
            continue
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, trackid)), z)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid)), x)


def main(instanc_size=511, num_threads=12):
    dataDir = '.'
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    for dataType in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(dataDir, dataType)

        annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
                                  set_crop_base_path, set_img_base_path, instanc_size) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
