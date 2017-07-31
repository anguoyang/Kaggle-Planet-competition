"""
best 0.901, CE 0.1088 - 33 epochs at current settings (EP1 0.833, ce 0.168)
using 72x72 images and random horizontal/ vertical flips at training

best LR for 32/32 - 0.05, 0.01(15), 0.005(20), 0.001(25)
best LR for 72/72 - 0.05, 0.01(24), 0.005(30), 0.001(40)

conv 32
drop 0.9
conv 64
max pool
drop 0.75
conv 128
max pool
drop 0.75
ga pool  -->>  1
conv 64 1x1    |
drop 0.5       |
conv 128       |
drop 0.5       |
ga pool  -->>  2 -->> concat -->> [17] -->> sigmoid CE
conv 64 1x1    |
drop 0.5       |
conv 128       |
drop 0.5       |
ga pool  -->>  3

"""
import sys
sys.path.append('C:\\Users\\LR2214\\AppData\\Local\\Programs\\Python\\Python35\\lib\\site-packages')
from utils import batch_generator, batch_generator_aug, reverse_dict
import tensorflow as tf
sess= tf.InteractiveSession()
import numpy as np
import pandas as pd
import os
import IPython


img_dim = 72
LAMBDA = 0.00000000 #0.000005 too high

# Construct the TF graph
x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, 4], name='x_placeholder')
y = tf.placeholder(tf.float32, shape=[None, 17], name='y_placeholder')
threshold = tf.placeholder(tf.float32, shape=[17], name='threshold')
phase = tf.placeholder(tf.bool, name='phase')
keep_prob = tf.placeholder(tf.float32, shape=[4])
learning_rate = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def global_avg_pool(x):
    dims_list = x.get_shape().as_list()
    return tf.nn.avg_pool(x, ksize=[1, dims_list[1], dims_list[2], 1], strides=[1, dims_list[1], dims_list[2], 1], padding='SAME')


def thresh_finder(x_val, y_val):

    np_thresh = np.zeros([17])
    best_thresh = np.zeros([17])

    for i in range(17):

        np_thresh.fill(0.2)
        best_f = 0
        
        for j in range(0, 35, 1):

            np_thresh[i] = j/100
            f = f_score.eval(feed_dict = {x:x_val, y:y_val, phase:1, keep_prob:dropout_vec_test, threshold:np_thresh})

            if f > best_f:
                best_thresh[i] = j/100
                best_f = f
    
    return best_thresh    

n_feats_1 = 32
n_feats_2 = 64
n_feats_3 = 128
n_feats_3_5 = 128

n_feats_3b = 16
n_feats_4 = 128
n_feats_4_5 = 128 
n_feats_4b = 16
n_feats_5 = 128
n_feats_5_5 = 128

neurons=512

W_conv1 = weight_variable([3, 3, 4, n_feats_1])
b_conv1 = bias_variable([n_feats_1])
conv1 = conv2d(x, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=phase))
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob=keep_prob[0])

W_conv2 = weight_variable([3, 3, n_feats_1, n_feats_2])
b_conv2 = bias_variable([n_feats_2])
conv2 = conv2d(h_conv1_drop, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=phase))
h_pool2 = max_pool_2x2(h_conv2) 
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob[1])

W_conv3 = weight_variable([3, 3, n_feats_2, n_feats_3])
b_conv3 = bias_variable([n_feats_3])
conv3 = conv2d(h_pool2_drop, W_conv3) + b_conv3
h_conv3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=phase))
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob=keep_prob[2])

W_conv3_5 = weight_variable([5, 5, n_feats_2, n_feats_3_5])
b_conv3_5 = bias_variable([n_feats_3_5])
conv3_5 = conv2d(h_pool2_drop, W_conv3_5) + b_conv3_5
h_conv3_5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv3_5, center=True, scale=True, is_training=phase))
h_pool3_5 = max_pool_2x2(h_conv3_5)
h_pool3_5_drop = tf.nn.dropout(h_pool3_5, keep_prob=keep_prob[2])


gap0 = global_avg_pool(h_pool3_drop)
gap0 = tf.reshape(gap0, [-1, gap0.get_shape().as_list()[-1]])
gap0 = tf.nn.l2_normalize(gap0, 1)

gap0_5 = global_avg_pool(h_pool3_5_drop)
gap0_5 = tf.reshape(gap0_5, [-1, gap0_5.get_shape().as_list()[-1]])
gap0_5 = tf.nn.l2_normalize(gap0_5, 1)

incep0 = tf.concat((h_pool3_drop, h_pool3_5_drop),3)

W_conv3b = weight_variable([1, 1, n_feats_3*2, n_feats_3b])
b_conv3b = bias_variable([n_feats_3b])
conv3b = conv2d(incep0, W_conv3b) + b_conv3b
h_conv3b = tf.nn.relu(tf.contrib.layers.batch_norm(conv3b, center=True, scale=True, is_training=phase))
h_pool3b_drop = tf.nn.dropout(h_conv3b, keep_prob=keep_prob[3])

W_conv4 = weight_variable([3, 3, n_feats_3b, n_feats_4])
b_conv4 = bias_variable([n_feats_4])
conv4 = conv2d(h_pool3b_drop, W_conv4) + b_conv4
h_conv4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv4, center=True, scale=True, is_training=phase))
h_pool4 = max_pool_2x2(h_conv4)
h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob=keep_prob[3])

W_conv4_5 = weight_variable([5, 5, n_feats_3b, n_feats_4_5])
b_conv4_5 = bias_variable([n_feats_4_5])
conv4_5 = conv2d(h_pool3b_drop, W_conv4_5) + b_conv4_5
h_conv4_5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv4_5, center=True, scale=True, is_training=phase))
h_pool4_5 = max_pool_2x2(h_conv4_5)
h_pool4_5_drop = tf.nn.dropout(h_pool4_5, keep_prob=keep_prob[3])

gap1 = global_avg_pool(h_pool4_drop)
gap1 = tf.reshape(gap1, [-1, gap1.get_shape().as_list()[-1]])
gap1 = tf.nn.l2_normalize(gap1, 1)

gap1_5 = global_avg_pool(h_pool4_5_drop)
gap1_5 = tf.reshape(gap1_5, [-1, gap1_5.get_shape().as_list()[-1]])
gap1_5 = tf.nn.l2_normalize(gap1_5, 1)

incep1 = tf.concat((h_pool4_drop, h_pool4_5_drop),3)


W_conv4b = weight_variable([1, 1, n_feats_4*2, n_feats_4b])
b_conv4b = bias_variable([n_feats_4b])
conv4b = conv2d(incep1, W_conv4b) + b_conv4b
h_conv4b = tf.nn.relu(tf.contrib.layers.batch_norm(conv4b, center=True, scale=True, is_training=phase))
h_pool4b_drop = tf.nn.dropout(h_conv4b, keep_prob=keep_prob[3])

W_conv5 = weight_variable([3, 3, n_feats_4b, n_feats_5])
b_conv5 = bias_variable([n_feats_5])
conv5 = conv2d(h_pool4b_drop, W_conv5) + b_conv5
h_conv5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv5, center=True, scale=True, is_training=phase))
h_pool5 = max_pool_2x2(h_conv5)
h_pool5_drop = tf.nn.dropout(h_pool5, keep_prob=keep_prob[3])

W_conv5_5 = weight_variable([5, 5, n_feats_4b, n_feats_5_5])
b_conv5_5 = bias_variable([n_feats_5_5])
conv5_5 = conv2d(h_pool4b_drop, W_conv5_5) + b_conv5_5
h_conv5_5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv5_5, center=True, scale=True, is_training=phase))
h_pool5_5 = max_pool_2x2(h_conv5_5)
h_pool5_5_drop = tf.nn.dropout(h_pool5_5, keep_prob=keep_prob[3])


gap2 = global_avg_pool(h_pool5_drop)
gap2 = tf.reshape(gap2, [-1, gap2.get_shape().as_list()[-1]])
gap2 = tf.nn.l2_normalize(gap2, 1)

gap2_5 = global_avg_pool(h_pool5_5_drop)
gap2_5 = tf.reshape(gap2_5, [-1, gap2_5.get_shape().as_list()[-1]])
gap2_5 = tf.nn.l2_normalize(gap2_5, 1)

conc_gap = tf.concat((gap0, gap0_5), 1)
conc_gap1 = tf.concat((conc_gap, gap1), 1)
conc_gap2 = tf.concat((conc_gap1, gap1_5), 1) 
conc_gap3 = tf.concat((conc_gap2, gap2), 1) 
conc_gap4 = tf.concat((conc_gap3, gap2_5), 1) 

W_fc1 = weight_variable([n_feats_5*6, neurons])
b_fc1 = bias_variable([neurons])

fc1 = tf.matmul(conc_gap4, W_fc1) + b_fc1
hfc1 = tf.nn.relu(tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=phase))
hfc1_drop = tf.nn.dropout(hfc1, keep_prob=keep_prob[0])

W_fc2 = weight_variable([neurons, y.get_shape().as_list()[1]])
b_fc2 = bias_variable([y.get_shape().as_list()[1]])

#Get the logits (shape [batch_size, classes])
logits_p = tf.matmul(hfc1_drop, W_fc2) + b_fc2

L2_reg = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_conv4)+ tf.nn.l2_loss(W_conv3b) + tf.nn.l2_loss(W_conv4b)

# Squash to range 0,1
sig = tf.nn.sigmoid(logits_p)

# Pick a threshold for positives
# np_thresh = np.zeros([17])
# np_thresh.fill(0.2)

#np_thresh = np.array([0.225, 0.175, 0.18, 0.105, 0.18, 0.205, 0.215, 0.26, 0.13, 0.155, 0.17, 0.06, 0.155, 0.19, 0.09, 0.115, 0.02])

np_thresh = np.array([0.22,  0.1,   0.2,   0.17,  0.18,  0.18,  0.21,  0.21,  0.12,  0.19,  0.17,  0.09, 0.17,  0.12,  0.16,  0.19,  0.09]) # augmented threshold

# Boolean matrices of predictions, and corresponding binary matrix for positives
predicted_one = sig >= threshold
predicted_zero = tf.logical_not(predicted_one)
preds = tf.cast(predicted_one, tf.float32)

# Calculate the cross entropy
ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p, labels=y)) + LAMBDA*L2_reg

# Create boolean matrices for the labels
is_label_one = tf.cast(y, tf.bool)
is_label_zero = tf.logical_not(is_label_one)

# Calculate TP,FP,TN, FN: each is a vector of length batch_size: a value for each image
true_pos = tf.reduce_sum(tf.cast(tf.logical_and(predicted_one,is_label_one),tf.float32),1) # when predicted 1 AND label is one
false_pos = tf.reduce_sum(tf.cast(tf.logical_and(predicted_one,is_label_zero),tf.float32),1) # when predicted one but label is zero
true_neg = tf.reduce_sum(tf.cast(tf.logical_and(predicted_zero,is_label_zero),tf.float32),1) # when predicted zero AND label is zero
false_neg = tf.reduce_sum(tf.cast(tf.logical_and(predicted_zero,is_label_one),tf.float32),1) # when predicted zero but label is 1

# Avoid division by zero
epsilon = tf.constant(1e-15)
precision = true_pos/(true_pos+false_pos+epsilon)
recall = true_pos/(true_pos+false_neg+epsilon)

# Take the mean across images of F score
f_score = tf.reduce_mean((5*precision*recall)/(4*precision+recall+epsilon))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(ce_loss)

# Start the session
sess.run(tf.global_variables_initializer())

batch_size = 48
epochs = 1000
test_epoch = 2
inf_epochs = [30, 44, 50, 54, 60, 70, 80, 90, 100] # must be multiple of test_epoch

# load in the 1-hot encoded labels
y_matrix = np.load('./train_labels.npy')

# directory with images
img_dir = './train-jpg_72'

# sort them by filename to correspond with y_matrix
unsorted_imgs = os.listdir(img_dir)
imgs = sorted(unsorted_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))

# Use this to limit overall data_size
# imgs = imgs2[:int(len(imgs2)*0.2)]
# y_matrix = y_matrix[:int(len(imgs2)*0.2),:]

# decide train/val split
train_no = 37477 # int(len(imgs)*0.9)  #37479# 

# lists of training and val images
train_imgs = imgs[:train_no]
val_imgs = imgs[train_no:]

# matrices of 1-hot encodings for train and val sets
train_y_matrix = y_matrix[:train_no,:]
val_y_matrix = y_matrix[train_no:,:]

#Pre-define the tensor of images to be used for val, and corresponding labels
test_batch_size = batch_size
val_batches = int(len(val_imgs)//test_batch_size+1)

# and for final testing
test_img_dir = './test-jpg_72'
unsorted_test_imgs = os.listdir(test_img_dir)
test_imgs = sorted(unsorted_test_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))

test_batches = int(len(test_imgs)//test_batch_size+1)

add_test_img_dir = './test-jpg-additional_72'
unsorted_add_test_imgs = os.listdir(add_test_img_dir)
add_test_imgs = sorted(unsorted_add_test_imgs,key=lambda f: int((f.split('_')[1]).split('.')[0]))
add_test_batches = int(len(add_test_imgs)//test_batch_size+1)

train_batches = int(train_no/batch_size+1)

# dropout_vec_train = np.array([0.9,0.75,0.75, 0.5])
dropout_vec_train = np.array([0.9,0.8,0.8, 0.8])
dropout_vec_test = np.array([1,1,1,1])

print('initialising... train size %g, val size %g' % (train_no, len(val_imgs)))
for epoch in range(1, epochs):

    # Decay the learning rate

    if epoch > 34:
        LR = 0.0001
    if epoch > 26:
        LR = 0.001
    if epoch > 20:
        LR = 0.005
    if epoch > 14:
        LR = 0.01
    else:
        LR = 0.05 

    print("epoch %d: training" % epoch)
    train_acc = 0
    train_loss = 0

    imgs_left = range(0,train_no,batch_size)

    for i in range(train_batches):

        if i%(train_batches//5) == 0:
            
            print('batch %g of %g' %(i, train_batches))

        start_trial = int(np.random.choice(imgs_left, 1, replace = False))
        # Remove that start from the list of those to be picked
        imgs_left = [i for i in imgs_left if i not in [start_trial]]

        # Generate a batch of images (randomly augmented)
        with tf.device('/cpu:0'):
            batch_x = batch_generator_aug(img_dir, train_imgs, batch_size, start_trial)
            batch_y = train_y_matrix[start_trial:start_trial+batch_size,:]
            
        # Train and accumulate the losses
        train_step.run(feed_dict = {x:batch_x, y:batch_y, phase:1, learning_rate:LR, keep_prob:dropout_vec_train})

        if epoch == 1 or epoch % test_epoch == 0:
            train_loss+= ce_loss.eval(feed_dict={x: batch_x, y: batch_y, phase:1, keep_prob:dropout_vec_test})
            train_acc += f_score.eval(feed_dict = {x:batch_x, y:batch_y, phase:1, keep_prob:dropout_vec_test, threshold: np_thresh})

    if epoch==1 or epoch%test_epoch==0:

        print('testing')

        val_loss = 0
        val_acc = 0

        aug_val_loss = 0
        aug_val_acc =0
        best_thresh = np.zeros([17])

        for i in range(val_batches):

            start_trial =i*test_batch_size

            with tf.device('/cpu:0'):
                # Genrate a batch from the validation set
                val_batch_x = batch_generator(img_dir, val_imgs, test_batch_size, start_trial)
                val_batch_y = val_y_matrix[start_trial:start_trial+test_batch_size,:]

                # Augment the batch of images
                default_im = []
                up_flip =[]
                left_flip = []
                both=[]

                for k in range(val_batch_x.shape[0]):
                    default_im.append(val_batch_x[k,:,:,:])
                    im_flip_up = np.flipud(val_batch_x[k,:,:,:])
                    up_flip.append(im_flip_up)
                    left_flip.append(np.fliplr(val_batch_x[k,:,:,:]))
                    both.append(np.fliplr(im_flip_up))

                aug_imgs_list = default_im + up_flip + left_flip + both
                aug_batch_x = np.stack(aug_imgs_list)

                aug_batch_y = np.tile(val_batch_y, (4,1))


            # val_loss += ce_loss.eval(feed_dict={x: val_batch_x, y: val_batch_y, phase:1, keep_prob:dropout_vec_test})
            # val_acc += f_score.eval(feed_dict = {x:val_batch_x, y:val_batch_y, phase:1, keep_prob:dropout_vec_test, threshold: np_thresh})

            # Get logits for original and augmented images
            aug_logits = logits_p.eval(feed_dict={x: aug_batch_x, y: aug_batch_y, phase:1, keep_prob:dropout_vec_test}) #[192,17]
            avg_logits = aug_logits[:val_batch_x.shape[0],:] + aug_logits[val_batch_x.shape[0]:2*val_batch_x.shape[0],:] + aug_logits[2*val_batch_x.shape[0]:3*val_batch_x.shape[0],:]+ aug_logits[3*val_batch_x.shape[0]:,:]
            
            # Take the average
            avg_logits /= 4

            # Use this average to calculate the loss
            aug_val_loss += ce_loss.eval(feed_dict={logits_p:avg_logits, y: val_batch_y, phase:1, keep_prob:dropout_vec_test})
            aug_val_acc += f_score.eval(feed_dict = {logits_p:avg_logits, y: val_batch_y, phase:1, keep_prob:dropout_vec_test, threshold: np_thresh})


            # if epoch in inf_epochs:
            #     print('finding thresholds, batch %g of %g' %(i, val_batches))
            #     best_thresh += thresh_finder(aug_batch_x, aug_batch_y)




        print("train CE: %g | train F2: %g" % (train_loss/train_batches, train_acc/train_batches))
        print('valid CE: %g | valid F2: %g' % (aug_val_loss/val_batches, aug_val_acc/val_batches))

        # if epoch in inf_epochs:

        #     best_thresh /= val_batches
        #     best_thresh = np.around(best_thresh,2)
        #     print('best thresholds on augmented val set:')
        #     print(best_thresh)
        #     print('giving f score on random batch: %f' % f_score.eval(feed_dict = {x:aug_batch_x, y:aug_batch_y, phase:1, keep_prob:dropout_vec_test, threshold:best_thresh}))
        #     print('compared to: %f' % f_score.eval(feed_dict = {x:aug_batch_x, y:aug_batch_y, phase:1, keep_prob:dropout_vec_test, threshold:np_thresh}))



    # Make a submission file
    if epoch in inf_epochs:
        print('writing submission file')
        filename = './submission_%s.csv' %epoch
        d = {'image_name' : [], 'tags' : []}

        for i in range(test_batches):
            start_trial = i*test_batch_size
            test_batch_x = batch_generator(test_img_dir, test_imgs, test_batch_size, start_trial)

            # Augment the test batch
            default_im = []
            up_flip =[]
            left_flip = []
            both=[]

            for k in range(test_batch_x.shape[0]):
                default_im.append(test_batch_x[k,:,:,:])
                im_flip_up = np.flipud(test_batch_x[k,:,:,:])
                up_flip.append(im_flip_up)
                left_flip.append(np.fliplr(test_batch_x[k,:,:,:]))
                both.append(np.fliplr(im_flip_up))

            aug_imgs_list = default_im + up_flip + left_flip + both
            aug_batch_x = np.stack(aug_imgs_list)

            # Get logit predictions for both default and augmented images, and take the average
            aug_logits = logits_p.eval(feed_dict={x: aug_batch_x, phase:1, keep_prob:dropout_vec_test}) #[192,17]
            avg_logits = aug_logits[:test_batch_x.shape[0],:] + aug_logits[test_batch_x.shape[0]:2*test_batch_x.shape[0],:] + aug_logits[2*test_batch_x.shape[0]:3*test_batch_x.shape[0],:]+ aug_logits[3*test_batch_x.shape[0]:,:]

            avg_logits /= 4

            # Use this average to make predictions
            p = preds.eval(feed_dict={logits_p:avg_logits, phase:1, keep_prob:dropout_vec_test, threshold:np_thresh})

            for j in range(p.shape[0]):
                indices = np.where(p[j,:] == 1)
                tags = ''
                for k in range(len(indices[0])):
                    tags += reverse_dict[indices[0][k]]
                    tags += ' '

                name ='test_%s'%(i*test_batch_size+j)
                d['image_name'].append(name)
                d['tags'].append(tags)

        print('half done')

        for i in range(add_test_batches):
            start_trial = i*test_batch_size
            test_batch_x = batch_generator(add_test_img_dir, add_test_imgs, test_batch_size, start_trial)
            
            default_im = []
            up_flip =[]
            left_flip = []
            both=[]

            for k in range(test_batch_x.shape[0]):
                default_im.append(test_batch_x[k,:,:,:])
                im_flip_up = np.flipud(test_batch_x[k,:,:,:])
                up_flip.append(im_flip_up)
                left_flip.append(np.fliplr(test_batch_x[k,:,:,:]))
                both.append(np.fliplr(im_flip_up))

            aug_imgs_list = default_im + up_flip + left_flip + both
            aug_batch_x = np.stack(aug_imgs_list)

            aug_logits = logits_p.eval(feed_dict={x: aug_batch_x, phase:1, keep_prob:dropout_vec_test}) #[192,17]
            avg_logits = aug_logits[:test_batch_x.shape[0],:] + aug_logits[test_batch_x.shape[0]:2*test_batch_x.shape[0],:] + aug_logits[2*test_batch_x.shape[0]:3*test_batch_x.shape[0],:]+ aug_logits[3*test_batch_x.shape[0]:,:]
            
            avg_logits /= 4

            p = preds.eval(feed_dict={logits_p:avg_logits, phase:1, keep_prob:dropout_vec_test, threshold:np_thresh})

            for j in range(p.shape[0]):
                indices = np.where(p[j,:] == 1)
                tags = ''
                for k in range(len(indices[0])):
                    tags += reverse_dict[indices[0][k]]
                    tags += ' '

                name ='file_%s'%(i*test_batch_size+j)
                d['image_name'].append(name)
                d['tags'].append(tags)

        df = pd.DataFrame(d)
        df.to_csv(filename, index=False)

        print('submission saved')
