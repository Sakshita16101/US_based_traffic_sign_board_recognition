STEPS-

1)main.py contains the driver code

2)Provide the path of "SEGMENTS" folder in "segments_folder_path" in main.py

3)Provide the path of ocr model in "model = load_model('Provide Model Path here')" in ocr.py

4)install all the requirements using pip install -r requirements.txt

5)Run main.py


Expected output for the given segment folder-

Anchor Positions: [0, 1, 2]
Character list: ['E', 'X', 'I']
Regular Expression: ^EXI.*
detected word: exit
Anchor Positions: [0, 1, 3]
Character list: ['L', 'I', 'I']
Regular Expression: ^LI.I.*
detected word: limit
Anchor Positions: [0, 1, 2]
Character list: ['S', 'P', 'E']
Regular Expression: ^SPE.*
detected word: speed

