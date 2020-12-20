import os 
import cv2
import imagehash
from PIL import Image
import numpy as np
import imutils
from mxnet import nd, image, gluon
from mxnet.io import ImageDetRecordIter

class Data():
    def __init__(self):
        self.labels = list()
        self.hashes = dict()
        self.hash_size = 16
        self.data_shape = None

    
    def get_images(self, directory, train=False):
        data, label = list(), list()
        for filename in os.listdir(directory):
            if os.path.isdir(directory + filename) and 'no' in filename:
                for fname in os.listdir(directory + filename):
                    img = cv2.imread(directory + filename + '/' + fname)
                    data.append(img)
                    label.append(0)

            if os.path.isdir(directory + filename) and 'yes' in filename:
                for fname in os.listdir(directory + filename):
                    img = cv2.imread(directory + filename + '/' + fname)
                    data.append(img)
                    label.append(1)
        
        if train:
            return self._data_augmentation(data, label)
        
        
        return np.array(data), np.array(label)


    def _data_augmentation(self, data, labels):
        ddata, llabels = list(), list()
        for img, l in zip(data, labels):
            # Rotate images by 90 degrees 
            ddata.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            llabels.append(l)

        for img, l in zip(data, labels):
            # Flip images vertically
            ddata.append(cv2.flip(img, 0))
            llabels.append(l)
        return np.concatenate((np.array(data), np.array(ddata)), axis=None), np.concatenate((np.array(labels), np.array(llabels)), axis=None)
                
            
    def check_duplicates(self, fname):
        with Image.open(fname) as img:
            temp_hash = imagehash.average_hash(img, self.hash_size)
            if temp_hash in self.hashes:
                return True
            else:
                self.hashes[temp_hash] = fname

    def _aug_transform(self, data, label):
        augs = image.CreateAugmenter(data_shape=self.data_shape, 
                                        rand_crop=0.5, 
                                        rand_mirror=True, 
                                        inter_method=10, 
                                        brightness=0.125, contrast=0.125, saturation=0.125, pca_noise=0.02)

        for aug in augs:
            data = aug(data)
        return data, label

    def data_loader(self, data, label, data_shape, transform=True, shuffle=True, batch_size=32):
        self.data_shape = data_shape
        for i, d in enumerate(data):
            d = d.astype('float32') / 255

            if transform:
                data[i], label = self._aug_transform(nd.array(d), label)        

            else:
                data[i]= image.imresize(nd.array(data[i]), w=data_shape[1], h=data_shape[2])
                
            data[i] = nd.transpose(data=nd.array(data[i]), axes=(2, 0, 1))

        dataset     = gluon.data.dataset.ArrayDataset(data, label)
        dataloader  = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def crop_imgs(self, imgs, add_pixels_value = 0):
        new_images = list()

        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Threshold the image then perform erosions and dilations to remove 
            # small regions of noise
            _, thresh   = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
            thresh      = cv2.erode(thresh, None, iterations=2)
            thresh      = cv2.dilate(thresh, None, iterations=2)

            # Find contours in threshold image
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key=cv2.contourArea)

            extLeft  = (c[c[:, :, 0].argmin()][0])
            extRight = c[c[:, :, 0].argmax()][0]
            extTop   = c[c[:, :, 1].argmin()][0]
            extBottom= c[c[:, :, 1].argmax()][0]

            new_img = img[extTop[1] - add_pixels_value : extBottom[1] + add_pixels_value, \
                        extLeft[0] - add_pixels_value : extRight[0] + add_pixels_value].copy()

            new_images.append(new_img)
        
        return np.array(new_images, dtype='object')
    
    def save_images(self, images, labels, directory):
        file_names = ['cropped{}.jpg'.format(i) for i in range(len(images))]
        for i, image in enumerate(images):
            if labels[i] == 1:
                d = directory + 'yes/'
                cv2.imwrite(d + file_names[i], image)
            else:
                d = directory + 'no/'
                cv2.imwrite(d + file_names[i], image)
            

