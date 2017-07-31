def batch_generator(img_dir, imgs, batch_size, start_trial):
    """
    Takes a list of image filenames, an integer batch size and a start index into the imgs and returns a 4-d tensor of
    batch size images with imgs indexed on the first dimension
    """

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # unsorted_imgs = os.listdir(img_dir)
    # imgs = sorted(unsorted_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))
    if start_trial + batch_size <= len(imgs):
    # if start_trial + batch_size <= len(imgs)*1000000000:

        img_batch = imgs[start_trial:start_trial+batch_size]
    else:
        img_batch = imgs[start_trial:]
        # rem = batch_size - len(img_batch)
        # for i in range(rem):
        #     img_batch.append(imgs[i])

    img_list = []
    for i in range(len(img_batch)):
        img_list.append(plt.imread(img_dir + '\\' + img_batch[i]))

    

    return np.stack(img_list)

# def inference():
#
#     test_img_dir = './test-jpg'
#     unsorted_imgs = os.listdir(test_img_dir)
#     imgs = sorted(unsorted_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))
#
#     batch_size = 100
#     test_batches = int(len(imgs) // batch_size)
#
#     for i in range(test_batches):
#         start_trial = i*batch_size
#         batch_x = batch_generator(test_img_dir,imgs,batch_size,start_trial)


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

# reverse_dict[10]
