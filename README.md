# ISG Reduction Model

This code is useful for estimating the level of phonetic reduction in speech data.  It was developed by Javier Vazquez and Nigel Ward of the Interactive Systems Group (ISG) of the University of Texas at El Paso (UTEP), in 2024-2025.  Publications describing the data, methods, and intended uses are available via https://www.cs.utep.edu/nigel/reduction/

To use this code, fork the repository and then follow the example usage below.  


## 1. To extract the HuBERT features:

`
from reduction_model import Reduction 
reduction = Reduction()
`

`
hubert_features = reduction.extract(['audio1.wav','audio2.wav','audio3.wav'])
`


## 2. Three options for training:

2a. Load the pretrained English model:

`
reduction.loadModel()
`

2b. Train your own model using our data. First download npy files
created at ISG, which are available at
www.cs.utep.edu/nigel/reduction/ , indo default_data.  Then:


`
reduction.default_fit()
`

2c. Train your own model using your own data. 

`
reduction.fit(X=hubert_features,y=['labels.txt'])
`

In general, once you have the HuBERT features available, you can
choose to train the model on any subset of these features for which
there are corresponding reduction labels. The labels format is a
tab-delimited file in the order of Channel, Start Time, End Time, and
Reduction Value. The Channel specifies Left or Right for Stereo audio
and None for Mono audio. The Start and End Time specify the timeframe
for utterances in seconds. The Reduction value specifies the
annotator's value for the specified region of speech. There are
examples in the default_data folder.

You can find a complete set of data for training from scratch in
 http://www.cs.utep.edu/nigel/reduction/annotations.zip (792KB).
 By default, you'll want to extract all the audio files and label files to
 default_data.


### 3. Two options for making preductions

3a. Now that you have a trained model, you can use it to predict the
reduction found at each frame.  Frames occur every 20ms.

`
frame_predictions = reduction.predict(hubert_features[0])
`

For this, you will need features extracted as described in Step 1.
The output will be the predicted reduction value vectors for each
track of the audio.

3b. Alternatively you can obtain per-region predictions using
 predict_utterance.  In most cases, these regions will likely be words
 or phrases. For this, a tab-delimited text file needs to be provided
 specifying the regions of interest.  This file will contain lines
 specifying the Channel, Start Time, and End Time, as described under
 2b.  If there is a fourth field, for the label, it will be ignored here.  **must there be at least a dummy value?**


`
utterance_predictions = reduction.predict_utterances(hubert_features,='utterance_timeframes.txt')
`

## Debugging Notes


### Python Setup

Install a recent version of python from www.python.org, and then 

`
_py -m pip install torch torchaudio numpy scikit-learn_
_py -m pip install sox soundfile matplotlib_    # optional
`

Note that you may need to use pip3 to get the modules in the right place, as in *pip3 install numpy*

Note that *py* is probably better than _python_, since newer,
depending on your configuration.


##Code Notes

After you get the code from [Javier Vazquez's
Github](https://github.com/javi-vaz/ISG_Reduction_Model) and extract
all the files, you will get a directory with a subdirectory called
REDUCTION.  Within that, you'll find the python files and a directory
called default_data.  It's easiest to invoke python from inside the
REDUCTION subdirectory.  You'll also see documentation, a pickled
model, and a tiny set of test data.

The downstream model (decision head) here is simple, as most of the
work is done by the HuBert features.  Thus, to predict the amount of
reduction in an audio, it first needs to be converted into HuBERT
features. This code provides easy access to the HuBERT model available
in PyTorch. In particular, it uses the HuBERT Base model which
extracts 12 layers of 768 features for every 20 ms of audio. The last
layer is the one used here as it had the best performance in pilot
tests. Note that, depending on whether the audio is mono or stereo,
the HuBERT model will produce features for one or two channels.

Note that reduction.extract() will return a list of torch tensors,
each of which has three dimensions: number-of-tracks (usually 2),
number of 20-ms frames, number of features per frame (which is 768).

Note that feature extraction takes some time, for example, perhaps 5
minutes to process a 10 minute file.

The API is designed to work with lists of files, not individual files,
let alone individual utterances or words.  This is because the
predictions are not trustworthy at the utterance or word level, so it
only makes sense to use this as part of a workflow collecting
statistics over substantial data, typically tens of minutes of dialog
across multiple audio files.

Note that at Github we include neither the audio files for training
nor the derived npy files; this is only because they are so large.
Accordingly the API is not very elegant.  However the code is flexible
enough to modify, with only a basic knowledge of python.


## To Test the Workflow


`
testFeats = reduction.extract(['tinytest/redu-enun-test.wav'])
testPreds = reduction.predict(testFeats[0])
`

(Of course you wouldn't usually test using a file that was included in the training data.)

Now you can visualize the predictions with, for example

`
    import matplotlib.pyplot as plt
    plt.plot(testPreds)
    plt.show()
`
then you can line it up the human labels, visualizable with Elan, to gauge quality. 
You can also compare the testPreds to the predictions we obtained, in redu-enum-test-predictions.npy

### Aside

As a point of interest, the audio files were originally taken from
 the DRAL corpus, which is downloadable from https://www.cs.utep.edu/nigel/dral/, or from 
 the [Linguistic Data Consortium](https://www.ldc.upenn.edu/) under Catalog number LDC2024S08

