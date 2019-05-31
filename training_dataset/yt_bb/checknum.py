import pandas as pd
import glob

col_names = ['youtube_id', 'timestamp_ms', 'class_id', 'class_name',
             'object_id', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']

sets = ['yt_bb_detection_validation', 'yt_bb_detection_train']

for subset in sets:
    df = pd.DataFrame.from_csv('./'+ subset +'.csv', header=None, index_col=False)
    df.columns = col_names
    vids = sorted(df['youtube_id'].unique())
    n_vids = len(vids)
    print('Total video in {}.csv is {:d}'.format(subset, n_vids))

    frame_download = glob.glob('./{}/*/*.jpg'.format(subset))
    frame_download = [frame.split('/')[-1] for frame in frame_download]
    frame_download = [frame[:frame.find('_')] for frame in frame_download]
    frame_download = [frame[:frame.find('_')] for frame in frame_download]
    frame_download = [frame[:frame.find('_')] for frame in frame_download]
    frame_download = sorted(set(frame_download))
    # print(frame_download)
    print('Total downloaded in {} is {:d}'.format(subset, len(frame_download)))


print('done')
