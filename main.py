import os
import random
from OCR import OCR
from Anchor_Generation import Anchor
from Prediction_Module import Regular_Expression
import re

segments_folder_path="/home/sakshita/Desktop/SEGMENTS"  #Path of the folder containing the segments

words=['CITYSPEEDLIMIT', 'MAXIMUM','SCHOOL','SPEED','EXIT','LIMIT']#Dictionay containing the possible texts that may be an input
d=['city speed limit','maximum','school','speed','exit','limit']#The same dictionay in lower case


"""""
This is the driver code and contains a method named final_result which calls methods of Anchor_Generation.py, OCR.py and Prediction_Module.py in order to get the regular
expression which needs to be searched in our dictionary named words.
"""
class Pipeline:
    def __init__(self,segments_folder_path):
        self.segments_folder_path=segments_folder_path

    def final_result(self):
        segments_folder_path=self.segments_folder_path
        for files in os.listdir(segments_folder_path):
            segments_path = os.path.join(segments_folder_path, files)
            # print(segments_path)
            anchor = Anchor(segments_path)
            position_list = anchor.Anchor()
            print("Anchor Positions:",position_list)

            character = OCR(position_list, segments_path)
            character_list=character.OpticalCharatcerRecognition()
            print("Character list:",character_list)

            # print("charatcer list",character_list)

            regex = Regular_Expression(character_list, position_list)
            RegEx=regex.regular_expression_generator()
            print("Regular Expression:",RegEx)



            for (i, j) in zip(words, d):
                # for i in words:
                if (re.search(str(RegEx), i)):
                
                    print("detected word:",j)

if __name__ == '__main__':
    out=Pipeline(segments_folder_path)
    out.final_result()






