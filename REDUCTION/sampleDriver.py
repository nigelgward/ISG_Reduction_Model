import numpy as np
from ../reduction_model import Reduction
from pathlib import Path

# runRedu.py
#  This is a little driver function to apply a reduction estimation model
#    to a list of audio files
#    and save the results to a set of csv files, one per track
#  This is to create features for use in the prosodic PCA workflow
#    in the Midlevel Prosodic Features Toolkit
#  Everything is hard-coded, to use English, and these specific files
#    but feel free to edit
#  call this from the command line with
#    py runRedu.py


def saveToFile(predictions, filename):
    # each frame here is 20 ms, so we write it twice to align with the 
    ofd = open(filename, 'w')
    timepoint = 0.10
    for item in predictions:
       ofd.write(str(timepoint) + str(item) + "\n")	
       timepoint = timepoint + 0.10
       ofd.write(str(timepoint) + str(item) + "\n")	
       timepoint = timepoint + 0.10

def createRedName(filename, trackChar):
    # create a reduction-features file
    basename = Path(filename).stem
    return basename + "-" + trackChar + '.red'


reduction = Reduction()
reduction.loadModel()

# derived from en-social/social.tl
filelist = ['/cygdrive/c/nigel/en-social/utep00.au', \
            '/cygdrive/c/nigel/en-social/utep04.au', \
            '/cygdrive/c/nigel/en-social/utep05.au', \
            '/cygdrive/c/nigel/en-social/utep07.au', \
            '/cygdrive/c/nigel/en-social/utep08.au', \
            '/cygdrive/c/nigel/en-social/utep21.au']

for file in filelist:
   feats = reduction.extract([file])
   preds = reduction.predict(feats[0])
   saveToFile(preds[0], createRedName(file, 'l'))
   saveToFile(preds[1], createRedName(file, 'r'))
