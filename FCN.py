#!/home/t630/anaconda3/bin python
from __future__ import print_function
import tensorflow as tf
import numpy as np
###import argparse

import TensorflowUtils as utils
#import read_MITSceneParsingData as scene_parsing
import read_Data as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

f_fcn=open('/model_bak/log/log_fcn','w+')

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 2, "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

#tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
   
tf.flags.DEFINE_string("data_dir", "/data/CardiacMRIs_folder/", "path to dataset")

tf.flags.DEFINE_float("learning_rate", "0", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224
#IMAGE_SIZE = 256


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image  
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            print('kind ==conv')
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            print('kernels')
            print(kernels)
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            print('bias')
            print(bias)
            current = utils.conv2d_basic(current, kernels, bias)
            print('current')
            print(current)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1)) 
    print("old mean_pixel======")
    print(mean_pixel)


    #scene_parsing.mean_images(args, ext='.png')    #old model 

    #==========================
    weights = np.squeeze(model_data['layers'])
    print("====weights==info====")
    print(weights)
    print("====image==info====")
    print(image.shape)



    #old======================= 
    #processed_image = utils.process_image(image, mean_pixel)
    #print(processed_image)
    #============================================================
    print("====scene_parsing==mean_images====")
    args = scene_parsing.parse_args()
    print(scene_parsing.mean_images(args, ext='.png'))

    new_mean_pixel = scene_parsing.mean_images(args, ext='.png')
    
    print('new_mean_pixel')
    print(new_mean_pixel)

    new_std_pixel = scene_parsing.std_images(args, ext='.png')
    print('new_std_pixel')
    print(new_std_pixel)

    processed_image = scene_parsing.Z_ScoreNormalization(image,new_mean_pixel,new_std_pixel)
    print('processed_image1')
    print(processed_image)
    #new=======================================
    print('before processed image')
    processed_image1 = utils.process_image(image, new_mean_pixel)   ##  暂时注掉 old mean nomalization method
    print('==begin processed image==')
    print(processed_image1)
    print('end  processed image')
    #===========================================================

    with tf.variable_scope("inference"):
        print('=====inference======')
        image_net = vgg_net(weights, processed_image)
        print('==inference==1=image_net=1=====')
        print(image_net)          
        conv_final_layer = image_net["conv5_3"]               
        print('=====inference=1==conv_final_layer===')
        print(conv_final_layer)          
        pool5 = utils.max_pool_2x2(conv_final_layer)         
        print('=====inference=3===pool5==')
        print(pool5)  
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        print('=====inference=4===W6=')
        print(W6)  
                  
        b6 = utils.bias_variable([4096], name="b6")
        print('=====inference=10==b6===')
        print(b6)          
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        print('=====inference=11===conv6==')
        print(conv6)          
        relu6 = tf.nn.relu(conv6, name="relu6")
        print('=====inference=11===relu6==')
        print(relu6)       
        if FLAGS.debug:
            utils.add_activation_summary(relu6)                   
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob) 
        print('=====inference=11===relu_dropout6==')
        print(relu_dropout6)
       
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        print('=====inference=20=====')
        print(W7)
        print(b7)
                  
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)         
        relu7 = tf.nn.relu(conv7, name="relu7")
        print('=====inference=21====')
        print(conv7)
        print(relu7)
                  
        if FLAGS.debug:
            utils.add_activation_summary(relu7)                  
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob) # ?????
        print('=====inference=22====')
        print(relu_dropout7)
        
        
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8") 
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        print('=====inference==3====')
        print(W8)
        print(b8)

                  
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        print('=====inference==31====')
        print(conv8)
         
        
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()                    
        print('=====inference==33==deconv_shape1==')
        print(deconv_shape1)
        
        W_t1 = utils.weight_variable( [4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1" ) 
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        print('=====inference==34====')
        print(W_t1)
        print(b_t1)         

                  
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")  
        print('=====inference==35====')
        print(conv_t1)
        print(fuse_1)
        print(conv8)
                  

        
        deconv_shape2 = image_net["pool3"].get_shape()
        print('=====inference==34==deconv_shape1==')
        print(deconv_shape2)
                  
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        print('=====inference==34==deconv_shape1==')
        print(W_t2)
        print(b_t2)    
                  
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
        print('=====inference==34==deconv_shape1==')
        print(conv_t2)
        print(fuse_2) 
    
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        print('=====inference==345==deconv_shape1==')
        print(shape)  
        print(deconv_shape3)
                  
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        print('=====inference==346==deconv_shape1==')
        print(W_t3)  
        print(b_t3)
                  
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        print('=====inference==347==deconv_shape1==')
        print(conv_t3)  
        #print(output_shape)


        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        print('=====inference==3456==deconv_shape1==')
        print(annotation_pred)
                  
    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    print('===FCN==train==')
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    print("=======FLAGS.learning_rate=======")
    print(FLAGS.learning_rate)
    print("=======optimizer=======")
    print(optimizer)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    print("=======grads==loss_val=====")
    print(loss_val)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)
    print("====main====tf.summary.scalar...==loss::====" )
    print(loss )
    print("====main====tf.summary.scalar...==loss==annotation:====" )
    print(annotation )
    
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    
    ######################################################
    ######################################################
    #train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)       # old  "reader dataset"
    print("fcn--->========readData========")
    args = scene_parsing.parse_args()
    train_records, valid_records = scene_parsing.readData(args)  
    print(len(train_records))   #260
    print(len(valid_records))   #545

    print("fcn--->Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        print("fcn---->FLAGS.mode == train ")
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        print("fcn---->=train_dataset_reader----important--")
        print(train_dataset_reader)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    print("fcn---->validation_dataset_reader-----important-- ")
    print(validation_dataset_reader)
    
    print("fcn======begin sess ========")
    sess = tf.Session()
    print("fcn======end sess ========")
    
    print("fcn--->Setting up Saver...")
    saver = tf.train.Saver()
    print("fcn--->Setting up Saver...end==")
    
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    print("fcn--->summary_writer...end==")
    
    sess.run(tf.global_variables_initializer())
    print("fcn--->sess.run...end==")
    print("fcn--->sess.run...end==")
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    print("fcn--->ckpt...end==")
    #print("fcn--->ckpt...end=="+ ckpt )
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("fcn--->Model restored...")
        print("fcn--->ckpt.....Model restored...")
        
    print("fcn--->before FLAGS.mode ==train==")
    if FLAGS.mode == "train":
        print("fcn--->enter FLAGS.mode ==train==")
        for itr in xrange(MAX_ITERATION):
            print("fcn--->enter FLAGS.mode ==train====MAX_ITERATION==")
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            
            print("fcn--->enter FLAGS.mode ==train====MAX_ITERATION==train_images==")
            print(train_images)       
            print(np.max(train_images))
            print(train_images.shape)
            print(train_images.dtype.name)
           
            print("fcn--->enter FLAGS.mode ==train====MAX_ITERATION==train_annotations==")
            print(train_annotations)
            print(np.max(train_annotations))
            print(train_annotations.shape)
            print(train_annotations.dtype.name)     #uint8  int32
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            print("fcn--->enter FLAGS.mode ==train====MAX_ITERATION==feed_dict==train_annotations")
            print(train_annotations)
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)
            
            
    elif FLAGS.mode == "test":               
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

if __name__ == "__main__":
    tf.app.run()
