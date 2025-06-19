Supplementary Notes

#	      	      ISG Reduction Model

This document is a baby-steps explanation of how to use Javier
 Vazquez's ISG_Reduction_Model, released at
 https://github.com/javi-vaz/ISG_Reduction_Model .  It was tested
 using bash in cygwin, and python 3.13.

<!---------------------------------------------------->
## Download the Code and Data, and start Python 

### Install a recent version of python from www.python.org, and then 

* _py -m pip install torch torchaudio numpy scikit-learn_
* _py -m pip install sox soundfile matplotlib_    # optional

   note that you may need to use pip3 to get the modules in the right place, as in *pip3 install numpy*

### Get the code from [Javier Vazquez's Github](https://github.com/javi-vaz/ISG_Reduction_Model);  for example, you might click on Code then "download zip".

  Then extract all the files.

  This will give you a directory with a subdirectory called REDUCTION.  Within that, you'll find the python files and a directory called default_data

### Start  python

- cd to the REDUCTION directory inside the ISG_Reduction_Model
- py       # probably not "python", depending on your configuration
- from reduction_model import Reduction
- reduction = Reduction()   # create the empty model 

  At this point you have a choice.  If you only care about English and are in a hurry, start with the Precomputed Features.  If you also want a Spanish reduction detector, or want to do everything from scratch, pick the From Audio option.


<!---------------------------------------------------->
## Use when Training on Precomputed Features (Option 1)

### Download the pre-computed Hubert features from  http://www.cs.utep.edu/nigel/reduction/redu-en-hubert-npy.zip

- extract and put all 5 npy files in the *default_data* folder

### Train the model 

- reduction.default_fit()

### Test the workflow 
- feats = np.load('default_dat/EN_006.npy')
- predictions = reduction.predict(feats)

  (Of course you wouldn't usually test using a file that was included in the training data.)

  Now we can visualize the predictions with, for example

    import matplotlib.pyplot as plt
    plt.plot(predictions[0])          # the whole left track
    plt.show()
    plt.clf()
    xpoints = np.arange(0.0, 10.0, 0.20)
    plt.plot(xpoints, frame_predictions[1][0:50])  # first 10 seconds of the right track

    *************how to test/verify**************


### Apply it to a new set of files

testFeats = reduction.extract(['default_data/redu-enun-test.wav'])
testPreds = reduction.predict(testFeats[0])
plt.plot(testPreds)
plt.show()
    then line it up with the Elan file, to gauge quality


   more often, you'll apply it to an entire sequence of files 
hubert_features = reduction.extract(['default_data/EN_006.wav','default_data/EN_007.wav', 'default_data/EN_013.wav','default_data/EN_033.wav','default_data/EN_043.wav'])

this will return a list of torch tensors, each of which has three dimensions: 
number-of-tracks (usually 2), number of 20-ms frames, number of features per frame (which is 768)

this computation takes some time, for example, 5 minutes to process a 10 minute file.



<!---------------------------------------------------->
## Use when training From Audio (Option 2)

### Prepare to compute the Hubert features from the audio files

-  download the annotations and audio files from http://www.cs.utep.edu/nigel/reduction/annotations.zip (792KB)
- extract this
- you will then need to ensure that both audio and label files are in default-data
-- if English, default_data already includes the annotations themselves, EN_006.txt, etc, so you'll just need to copy the newly extracted labelable-audios/EN*wav to default_data
-- if Spanish, similarly copy over ES*wav, and in addition ES*txt from the newly-extracted annotations-cao directory, all into default_data	

- as a point of interest, the audio files were originally taken from
 the DRAL corpus, which is downloadable from https://www.cs.utep.edu/nigel/dral/, or from 
 the [Linguistic Data Consortium](https://www.ldc.upenn.edu/) under Catalog number LDC2024S08



- first we compute the features

hubert_features = reduction.extract(['default_data/EN_006.wav','default_data/EN_007.wav', 'default_data/EN_013.wav','default_data/EN_033.wav','default_data/EN_043.wav'])

   as a side note, the dimensions of the result are: number-of-files, 

  this may take 30 to 50 seconds per stero minute of audio


   After completion, if you're planning to do further experiments, you can  save the results, for example with
   
np.save('default_data/EN_006.npy', hubert_features[0]) 

   These npy files are large: about 20 MB per minute of audio


### Train The Model 


reduction.fit(X=hubert_features, y=['default_data/EN_006.txt', 'default_data/EN_007.txt', 'default_data/EN_013.txt', 'default_data/EN_033.txt', 'default_data/EN_043.txt'])


<!---------------------------------------------------->
### Apply the model


This is generally done file-by-file.  For example, to get the
predictions for file EN_006.wav, after the extract() above,

frame_predictions = reduction.predict(hubert_features[0]) 

This is just an example.  In general you wouldn't test using a file that was included in the training data.  In general your workflow is
  newfile_features=reduction.extract(['data/new_audio_file.wav'])
  new_predictions = reduction.fit(X=newfile_features)

### alternatively, you can get per-region predictions using
region_preds = reduction.predict_utterances(hubert_features[0], default_data/EN_006.txt')

   where the second argument is used just to specify the regions, not the labels


### to support use for PCA: 
train model
for all audio files listed in a .tl  (tracklist) file
   apply the model
   write a reduction-values file, with timepoints and values every 20 ms
   e.g. utep00.aur-redu.csv
         0.010 2.9
         0.030 2.1
         0.050 1.7
         and so on 	 
then in makeTrackMonster,
   add a code for this (and create a .fss file with that code)
   and add a function called lookupRedu() similar to lookupOrComputePitch()


