import numpy as np
from reduction_model import Reduction
from pathlib import Path

# sampleDriver.py
#  This is a little driver function to apply a reduction estimation model
#    to a list of audio files (hardcoded below)
#    and save the results to a set of csv files, one per track
#  This is to create features for use in the prosodic PCA workflow
#    in the Midlevel Prosodic Features Toolkit
#  Everything is hard-coded, to use English and these specific files,
#    so feel free to edit
#  call this from the command line with
#    py sampleDriver.py


def saveToFile(predictions, filename):
    # each frame here is 20 ms, so we write it twice to align with the 10ms frame files
    ofd = open(filename, 'w')
    twentyMsFrameNum = 0
    for item in predictions:
        timestamp1 = 0.010 + 0.020 * twentyMsFrameNum 
        timestamp2 = 0.020 + 0.020 * twentyMsFrameNum 
        ofd.write(f"{timestamp1:.3f} \t {item:.2f} \n")	
        ofd.write(f"{timestamp2:.3f} \t {item:.2f} \n")	
        twentyMsFrameNum = twentyMsFrameNum + 1 

# create a reduction-features filename; for now, just in the current directory 
def createRedName(filename, trackChar):
    return Path(filename).stem + "-" + trackChar + '.txt'

reduction = Reduction()
reduction.loadModel()

# derived from en-social/social.tl
# filenames relative to c:/nigel 

filelist = ['c:/nigel/en-social/utep00.au', \
            'c:/nigel/en-social/utep04.au', \
            'c:/nigel/en-social/utep05.au', \
            'c:/nigel/en-social/utep07.au', \
            'c:/nigel/en-social/utep08.au', \
            'c:/nigel/en-social/utep21.au']
filelist = ['c:/nigel/ml2/midlevel/flowtest/21d.au']
filelist = ['tinytest/redu-enun-test.wav']

for file in filelist:
    print(f"model class: extracting features for {file}")
    feats = reduction.extract([file])
    preds = reduction.predict(feats[0])

    print(f"saving to file {createRedName(file, 'l')}")
    saveToFile(preds[0], createRedName(file, 'l'))
    # uncomment these two if it's a stereo file 
    print(f"saving to file {createRedName(file, 'r')}")
    saveToFile(preds[1], createRedName(file, 'r'))
