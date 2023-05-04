import os
import random


""""
This code generates anchor charatcers which are random numbers produced in the range of length of the word.The minimum number of anchors to be passed is 3(when 40% of length
of text is less than 3) in rest of the cases characters equal(in number) to 40% of the length of the text has to be passed.

"""
class Anchor:
    def __init__(self, segments_path):
        self.segments_path=segments_path

    def Anchor(self):
        segments_path= self.segments_path
        randomlist = [] #List which will contain the reandom numbers corresponding to anchor position
        for files in os.listdir(segments_path):
            no_of_segments = len(os.listdir(segments_path))  # No of segments generated from our cutout
            no_of_charatcers_to_pass = (0.4 * no_of_segments)# No of segments to be passed i.e equal in number to 40% of length of word
            if (no_of_charatcers_to_pass < 3):# If 40% of total length of word is less than 3 number of characters to be passed = 3
                no_of_charatcers_to_pass = 3
            randomlist = random.sample(range(0, no_of_segments - 1), int(no_of_charatcers_to_pass))
            randomlist.sort()
            return randomlist# Returns a list conatining the anchor numbers








