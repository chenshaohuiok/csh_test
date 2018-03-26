"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import cv2


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        print (  'Batch Dataset enter Reader_read_images ' )
        self.__channels = True
              
        print( '===_read_images==print=self.files====' )
        print( self.files )
        print( '===_read_images==print=self.iamges====' )
        print (self.images)
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        #self.images = np.array(
        #    [np.expand_dims(self._transform(filename['image']),axis=3) for filename in self.files])
        print( '===_read_images==print=self.images.np.array====' )
           
        print (self.images.ndim)
        print (self.images.shape)
        print (self.images.size)
        print (self.images.itemsize)         # ?? what
        print (self.images.dtype.name)
        print (self.images.mean())
        print (self.images.max())
        print (self.images.min())
        print('================')
        print(np.max(self.images))
        print(np.min(self.images))
        #self.images = np.array(
        #    [np.expand_dims(self._transform(filename['image']),axis=3) for filename in self.files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform_anno(filename['annotation']), axis=3) for filename in self.files])

        print (self.annotations.ndim)
        print (self.annotations.shape)
        print (self.annotations.size)
        print (self.annotations.itemsize)         # ?? what
        print (self.annotations.dtype.name)

        print ('Batch Dataset==self.images.shape')
        print (self.images.shape)
        print ('Batch Dataset==self.annotations.shape')
        print (self.annotations.shape)
        print ('Batch Dataset==self.images[10:20]==')
        print (self.images[10:20])
        #print (self.images.reshape(224, 224))
        print ('Batch Dataset==self.annotations')
        ##print (self.annotations)
        print ('Batch Dataset==_read_images===end===<<')

    def _transform(self, filename):

        print ('Batch Dataset==_transform==filename')
        print (filename)
        image = misc.imread(filename)
        #for i in range(256):
        #    for j in range(256):
        #        image[i, j] = np.float32(0.0)
        image = image.astype(np.float32)
        print (image)
        print(image.shape)   ## 256 X256
        print(np.max(image))
        print (image.dtype.name)
        print (self.__channels)
        print (len(image.shape))
        print (image.mean())
        print (image.max())
        print (image.min())
        if self.__channels and len(image.shape) < 3:        # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

            print ('Batch Dataset==_transform==self.__channels and len(image.shape) < 3==')
            print (image)          ####正确34684
            print (np.max(image))
            print (image.dtype.name)
            print(image.shape)
            image=np.transpose(image, (1, 2, 0))
            print (image.shape)
            print (np.max(image))
            print (image.dtype.name)
                  
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            print ('Batch Dataset====_transform====resize_size===before==')
            print (resize_size)
            print(image.dtype.name)
            print(image.shape)
            resize_image = cv2.resize(image, (resize_size, resize_size))
            print ('Batch Dataset==_transform==self.__channels and len(image.shape) < 3==resize_image')
            print(resize_image.dtype.name)
            resize_image=resize_image.astype(np.float32)
            
            print(resize_image)
            print(resize_image.shape)
            print(np.max(resize_image))
            print ('Batch Dataset==_transform===resize_image=if==')
        else:
            print ('Batch Dataset==_transform===resize_image=else==')
            print(image.dtype.name)
            resize_image = image
            print(resize_image.dtype.name)
            
            
        print ('Batch Dataset==_transform===resize_image=')
        
        print (resize_image)
        print ('Batch Dataset==_transform===resize_image.dtype.name=')
        print (resize_image.dtype.name )
        return np.array(resize_image)

    def _transform_anno(self, filename):

        print ('Batch Dataset==_transform_anno==filename')
        print (filename)
        image = misc.imread(filename)
        image = image.astype(np.uint32)
        print (image)
        print(image.shape)
        print(np.max(image))
        print (image.dtype.name)
        print (image)
        print(np.max(image))
        print (self.__channels)
        print (len(image.shape))
        
        if self.__channels == False and len(image.shape) < 3:
            print ('Batch Dataset==_transform_anno==self.__channels ==false==binary_image')
            image = image.astype(np.uint32)
            image_bin = self._bin_transform(image)
            print (image_bin)
            print (image_bin.dtype.name)
            print(np.max(image_bin))            ###   正确  1
            image = image_bin
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            print ('resize_size')
            print (resize_size)
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
            print ('Batch Dataset==_transform_anno===resize_image=if==')
        else:
            resize_image = image
            
            print ('Batch Dataset==_transform_anno===resize_image=else==')
        print ('Batch Dataset==_transform_anno===resize_image=')
        
        print (resize_image)
        print ('Batch Dataset==_transform_anno===resize_image.dtype.name=')
        print (resize_image.dtype.name )
        return np.array(resize_image)

    
    def _bin_transform(self,image):
        print ('Batch Dataset-->bin_transform_image-->')
        img = image
        print(img.shape)
        height, width = img.shape[:2]
        rows = height 
        cols = width 
        print(rows)
        print(cols)
        for i in range(rows):
            for j in range(cols):
                if (img[i,j]>=2):
                    img[i,j]=1
                else:
                    img[i,j]=0
        print('_bin_transform')
        print(img)
        print(np.max(img))         
        return img



    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        print(" Batch Dataset Reader----> next_batch")
        start = self.batch_offset
        self.batch_offset += batch_size
        print("Batch Dataset Reader----> next_batch......")
        self.images = self.images.astype(np.float32)
        print(self.images)
        print(self.images.dtype.name)
        print(self.images.shape)       ## 244,244,3
        print(np.max(self.images))     ## max 255
        print(self.images.shape[0])
        print(self.images.shape[1])
        print(self.images.shape[2])
        print(self.batch_offset)        ##2
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            
            print("Batch Dataset Reader----> next_batch->self.images[perm]....===")
            print (self.images)
            print(self.images.shape)
            print(self.images.dtype.name)
            print(np.max(self.images))           
            self.annotations = self.annotations[perm]
            print("Batch Dataset Reader----> next_batch->self.annotations[perm]...====.")
            print (self.annotations)
            print(self.annotations.shape)
            print(self.annotations.shape[0])
            print(self.annotations.shape[1])
            print(self.annotations.shape[2])
            print(np.max(self.annotations))  
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
