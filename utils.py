def batch_generator(img_dir, imgs, batch_size, start_trial):
    """
    Takes a list of image filenames, an integer batch size and a start index into the imgs and returns a 4-d tensor of
    batch size images with imgs indexed on the first dimension
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if start_trial + batch_size <= len(imgs):
        img_batch = imgs[start_trial:start_trial+batch_size]
    else:
        img_batch = imgs[start_trial:]

    img_list = []
    for i in range(len(img_batch)):
        img_list.append(plt.imread(img_dir + '\\' + img_batch[i])[:,:,:3])

    return np.stack(img_list)

string_labels = ['haze',
 'primary',
 'agriculture',
 'clear',
 'water',
 'habitation',
 'road',
 'cultivation',
 'slash_burn',
 'cloudy',
 'partly_cloudy',
 'conventional_mine',
 'bare_ground',
 'artisinal_mine',
 'blooming',
 'selective_logging',
 'blow_down']

label_dict = {}

count = 0
for i in string_labels:
    label_dict[i] = count
    count+=1

reverse_dict = {}

for tag in string_labels:
    idx = label_dict[tag]
    reverse_dict[idx] = tag

weather_labels = ['clear',
 'haze',
 'partly_cloudy',
 'cloudy']

weather_dict = {}

count = 0
for i in weather_labels:
    weather_dict[i] = count
    count+=1

reverse_weather_dict = {}

for tag in weather_labels:
    idx = weather_dict[tag]
    reverse_weather_dict[idx] = tag

other_labels = ['primary',
'agriculture',
'water',
'habitation',
'road',
'cultivation',
'slash_burn',
'conventional_mine',
'bare_ground',
'artisinal_mine',
'blooming',
'selective_logging',
'blow_down']

other_dict = {}

count = 0
for i in other_labels:
    other_dict[i] = count
    count+=1

reverse_other_dict = {}

for tag in other_labels:
    idx = other_dict[tag]
    reverse_other_dict[idx] = tag


#### Make dictionary for full labels ie multi word strings (weaher tags excluded)
# 278 possibilities

import pandas as pd

labels_df = pd.read_csv('train_v2.csv')
complete_labels = []
reverse_complete_dict ={}
complete_dict = {}

count = 0
for tag in labels_df['tags']:
    split_tag = tag.split(' ')
    multi_labels =  [i for i in split_tag if i in other_labels]
    new_tag =''
    for k in range(len(multi_labels)):
        new_tag +=multi_labels[k]
        
        if k != len(multi_labels)-1:
            new_tag+= ' '

    if new_tag not in complete_labels and new_tag != '':
        complete_labels.append(new_tag)
        complete_dict[new_tag] = count
        reverse_complete_dict[count] = new_tag
        count+=1

def batch_generator_aug(img_dir, imgs, batch_size, start_trial):
    """
    Takes a list of image filenames, an integer batch size and a start index into the imgs and returns a 4-d tensor of
    batch size images with imgs indexed on the first dimension
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import IPython

    if start_trial + batch_size <= len(imgs):
        img_batch = imgs[start_trial:start_trial+batch_size]
    else:
        img_batch = imgs[start_trial:]

    img_list = []
    for i in range(len(img_batch)):
        im = plt.imread(img_dir + '\\' + img_batch[i])
        if np.random.rand() <0.5:
            im = np.flipud(im)
        if np.random.rand() >0.5:
            im = np.fliplr(im)
        # im_aug = tf.image.random_flip_left_right(im)
        # im_aug = tf.image.random_flip_up_down(im_aug)
        img_list.append(im[:,:,:3]) # leave off last dim

        # angles = np.random.randint(0,360,size=batch_size)
        # imgs = np.stack(img_list)
        # aug_imgs = tf.contrib.image.rotate(imgs, angles, interpolation='NEAREST')

        # IPython.embed()

    return np.stack(img_list)
