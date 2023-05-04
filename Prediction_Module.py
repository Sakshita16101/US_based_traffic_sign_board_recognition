from Anchor_Generation import Anchor
from OCR import OCR

# anchor= Anchor("/home/sakshita/Desktop/SEGMENTS/limit")
# position_list=anchor.Anchor()
#
# characters=OCR(position_list,"/home/sakshita/Desktop/SEGMENTS/limit")
# character_list=characters.OpticalCharatcerRecognition()

"""""
This code accepts anchor positions and corresponding character characters in the form of list and generates a regular expression correspond to them. This regular expression 
then needs to be searched in our dictionary to get the correct result
"""
class Regular_Expression:
    def __init__(self,character_list,position_list):
        self.character_list=character_list
        self.position_list=position_list

    def regular_expression_generator(self):
        character_list=self.character_list
        position_list=self.position_list
        count = 0
        Regex = '^'
        for (i, j) in zip(position_list, character_list):
            if (count == int(i)):
                Regex = Regex + j
                count = count + 1
            else:
                while (count < int(i)):
                    Regex = Regex + '.'
                    count = count + 1
                    if (count == int(i)):
                        Regex = Regex + j
                        count = count + 1
        Regex = Regex + '.*'
        return Regex

#
# if __name__ == '__main__':
#     A=Regular_Expression(character_list,position_list)
#     print(A.regular_expression_generator())
