__author__ = 'lenovo'
import argparse
import os
import fnmatch
import cv2
import numpy as np
from numpy import *
import scipy.misc as misc

f_train=open('/model_bak/log/log_train','w+')
f_validation=open('/model_bak/log/log_validation','w+')
#f_train.write('111111')
# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--root_dataset', default='/data/CardiacMRIs_folder')
    parser.add_argument(
        '--output_csv', default='{}.csv')
    return parser.parse_args()


# finder func
def find_images(root_dir, ext='.png'):
    images = []


    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            images.append(  os.path.join(root, filename)  )
    return images


### func mean_images
def mean_images(args, ext='.png'):
    print ('=read_data====enter mean_images===:')
    images_records = []
    img_size=256
    sum_r=0
    sum_g=0
    sum_b=0
    count=0

    for subset in ('training', 'validation') :
      
        path_images_subset = os.path.join(args.root_dataset, 'images', subset)
        #path_annotations_subset = os.path.join(args.root_dataset, 'annotations', subset)
        print ('path_images_subset:')
        print (path_images_subset)
        subset_images = find_images(path_images_subset)

        for image in subset_images:
            print ('for image in subset_images:')
            print (image)
            images_records.append( image  )

    print (images_records)

    for img_name in images_records:
        img_path=os.path.join( path_images_subset , img_name )
        
        #img=cv2.imread(img_path,1)
        #print(img.dtype.name)
        image = misc.imread(img_path)
        img = image.astype(np.float32)
        #img = img.astype(np.float32)
        print('Gray--img')
        print(img)
        print(img.dtype.name)
        print('img.shape')   #
        print(img.shape)     #(256, 256)(512, 512, 3) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        img_std= np.std(np.std(img,0,ddof=1),ddof=1) 
        print ('img_std')
        print (img_std)
   
        print('Gray--img.shape')

        sum_r=sum_r+img.mean()
        count=count+1
    print ('img_mean_count')
    print (count)
    sum_r=sum_r/count
    img_mean=sum_r
    return img_mean

##function images_sta
def std_images(args, ext='.png'):
    print ('=====enter std_images===:')
    images_records = []
    img_size=256
    count=0
    g_std=0

    for subset in ('training', 'validation') :
      
        path_images_subset = os.path.join(args.root_dataset, 'images', subset)
        #path_annotations_subset = os.path.join(args.root_dataset, 'annotations', subset)
        print ('path_images_subset:')
        print (path_images_subset)
        subset_images = find_images(path_images_subset)

        for image in subset_images:
            print ('for image in subset_images:')
            print (image)
            images_records.append( image  )

    print (images_records)

    for img_name in images_records:
        img_path=os.path.join( path_images_subset , img_name )
        
        #img=cv2.imread(img_path,1)
        image = misc.imread(img_path)
        img = image.astype(np.float32)
        g_std = g_std +np.std(img)
        count=count+1
    print ('img_mean_count')
    print (count)
    g_std=g_std/count
 
    print ('g_std')
    print (g_std)
    return g_std

### Z_ScoreNormalization function
def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    print (x)
    return x




# main func
def readData(args):
    training_records = []
    validation_records = []
    record = {}
    for subset in ('training', 'validation'):
        pairs = []
        #appendStr=''
        record = {}
        path_images_subset = os.path.join(args.root_dataset, 'images', subset)
        path_annotations_subset = os.path.join(args.root_dataset, 'annotations', subset)

        subset_images = find_images(path_images_subset)
        # loop over each image
        for image in subset_images:
            print ('for image in subset_images:')
            print (image)
            filename = (image.split("/")[-1])    #[0]
            print ('filename')
            print (filename)
            filename0 = (filename.split(".")[0])    #[0]
            print ('filename0')
            print (filename0)
            annotation = image.replace(path_images_subset, path_annotations_subset)
            # check if annotation exists
            if os.path.exists(annotation):
                record = {'annotation': annotation, 'image': image, 'filename': filename0 }
                pairs.append( record  )
              
        print(pairs,file=f_validation)
        print(   'Subset [{}] has [{}] samples.'.format(subset, len(pairs)  )   )

        if subset == 'training':
             print('training')
             training_records = pairs[:]
             output_csv = os.path.join(   args.root_dataset, args.output_csv.format(subset)  )
             #with open(output_csv, 'w') as f:
                #for pair in training_records:
                    #f.write(pair + '\n')
             print(training_records,file=f_train)
        else:
            print('validation')
            validation_records = pairs[:]
            output_csv = os.path.join(   args.root_dataset, args.output_csv.format(subset)  )
            #with open(output_csv, 'w') as f:
                #for pair in validation_records:
                    #f.write(pair + '\n')
            print(validation_records,file=f_validation)

    return training_records , validation_records

if __name__ == '__main__':
    args = parse_args()
    readData(args)
print('Done!')
