from pycocotools.coco import COCO
import cv2
import numpy as np

color_bar = np.random.randint(0, 255, (90, 3))

visual = True

dataDir = '.'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

for img_id in coco.imgs:
    img = coco.loadImgs(img_id)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    im = cv2.imread('{}/{}/{}'.format(dataDir, dataType, img['file_name']))
    for ann in anns:
        rect = ann['bbox']
        c = ann['category_id']
        if visual:
            pt1 = (int(rect[0]), int(rect[1]))
            pt2 = (int(rect[0]+rect[2]), int(rect[1]+rect[3]))
            cv2.rectangle(im, pt1, pt2, color_bar[c-1], 3)
    cv2.imshow('img', im)
    cv2.waitKey(200)
print('done')

