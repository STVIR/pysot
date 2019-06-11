from os.path import join, isdir
from os import mkdir
import glob
import xml.etree.ElementTree as ET
import json

js = {}
VID_base_path = './ILSVRC'
ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)

    if 'a' == sub_set:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
    else:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
    n_imgs = len(xmls)
    for f, xml in enumerate(xmls):
        print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')

        video = join(sub_set, xml.split('/')[-1].split('.')[0])

        for id, object_iter in enumerate(objects):
            bndbox = object_iter.find('bndbox')
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            frame = '%06d' % (0)
            obj = '%02d' % (id)
            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            js[video][obj][frame] = bbox

train = {k:v for (k,v) in js.items() if 'i/' not in k}
val = {k:v for (k,v) in js.items() if 'i/' in k}

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)
