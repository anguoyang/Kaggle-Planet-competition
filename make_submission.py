def make_submission(filename):
    from utils import batch_generator, reverse_dict, reverse_other_dict, reverse_weather_dict
    import pandas as pd
    
    # and for final testing
    test_img_dir = './test-jpg'
    unsorted_test_imgs = os.listdir(test_img_dir)
    test_imgs = sorted(unsorted_test_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))
    test_batch_size = 100
    test_batches = int(len(test_imgs)//test_batch_size+1)

    add_test_img_dir = './test-jpg-additional'
    unsorted_add_test_imgs = os.listdir(add_test_img_dir)
    add_test_imgs = sorted(unsorted_add_test_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))
    add_test_batches = int(len(add_test_imgs)//test_batch_size+1)


    d = {'image_name' : [], 'tags' : []}

    for i in range(test_batches):
        start_trial = i*test_batch_size
        test_batch_x = batch_generator(test_img_dir, test_imgs, test_batch_size, start_trial)
        weather_preds, _ = atmos_class(batch_x = test_batch_x, batch_y=0, training=False)
        multi_preds, _ = multi_class(batch_x = test_batch_x, batch_y=0, training=False)

        for j in range(weather_preds.shape[0]):
            weather_idx = np.where(weather_preds[j,:]==1)
            tags = reverse_weather_dict[weather_idx][0]

            if tags != 'cloudy':

                multi_indices = np.where(multi_preds[j,:] == 1)
                for k in range(len(multi_indices[0])):
                    tags += reverse_other_dict[multi_indices[0][k]]
                    tags += ' '

            name ='test_%s'%(i*test_batch_size+j)
            d['image_name'].append(name)
            d['tags'].append(tags)

    for i in range(add_test_batches):
        start_trial = i*test_batch_size
        test_batch_x = batch_generator(add_test_img_dir, add_test_imgs, test_batch_size, start_trial)
        weather_preds, _ = atmos_class(batch_x = test_batch_x, batch_y=0, training=False)
        multi_preds, _ = multi_class(batch_x = test_batch_x, batch_y=0, training=False)

        for j in range(weather_preds.shape[0]):
            weather_idx = np.where(weather_preds[j,:]==1)
            tags = reverse_weather_dict[weather_idx][0]

            if tags != 'cloudy':

                multi_indices = np.where(multi_preds[j,:] == 1)
                for k in range(len(multi_indices[0])):
                    tags += reverse_other_dict[multi_indices[0][k]]
                    tags += ' '

            name ='test_%s'%(i*test_batch_size+j)
            d['image_name'].append(name)
            d['tags'].append(tags)

    df = pd.DataFrame(d)
    df.to_csv(filename, index=False)
