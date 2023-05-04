from keras.models import load_model
import os
import cv2
from PIL import Image,  ImageEnhance
import numpy as np
from Anchor_Generation import Anchor


labels_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'A', 11: 'B', 12: 'C', 13: 'D',
               14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
               25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',}


model = load_model('/home/sakshita/modelrgb_dataset.h5')#Load our CNN model


"""""
This code accepts a position list which contains the anchor characters and does OCR only on the segments corresponding to those positions mentioned in the list. Finally
it returns a list containing the characters to those anchor positions.
"""
class OCR:
    def __init__(self,position_list,segments_path):
        self.position_lsit=position_list
        self.segments_path=segments_path

    def OpticalCharatcerRecognition(self):
        position_list =self.position_lsit
        segments_path=self.segments_path
        # print(path)
        character_list=[]
        # anchor_charatcers = Anchor(segments_path)
        for anchor in position_list:
            new_path = segments_path + "/" + str(anchor) + ".png"#Path corresponding to the path of the images
            test_image = cv2.imread(new_path) # Reading the segments
            test_image = cv2.resize(test_image, (10, 10), cv2.INTER_CUBIC)
            cutouts = Image.fromarray(test_image)
            img = ImageEnhance.Contrast(cutouts).enhance(1.5)

            test_image1 = np.asarray(img)
            test_image = test_image1.reshape(10, 10, 3) # Resizing the segments to a size of (10,10)

            # test_image=images[4000]
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            result = (np.argmax(result, axis=1))
            result_label = labels_dict[int(result)]
            character_list.append(result_label)
        return character_list # Return a list corresponding to the anchor numbers






# if __name__ == '__main__':
#     k = OCR(position_list,"/home/sakshita/Desktop/SEGMENTS/limit")
#     print(k.OpticalCharatcerRecognition())

