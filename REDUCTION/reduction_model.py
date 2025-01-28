import os
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from feature_extractor import FeatureExtractor


class Reduction:

    def __init__(self):
        self.regression_model = LinearRegression()
        self.hubert = FeatureExtractor()

    def extract(self,audios,save=False):
        """Extracts the 12th layer of HuBERT Base features for every audio in a given audio list.

        Args:
            audios - A list containing the path to wav files that will be processed using HuBERT.

        Returns:
            hubert_features - A dictionary containing the 12th layer of hubert features for every audio processed. 
        
        """
        hubert_features = {}
        for audio in audios:
            path = os.path.join(os.getcwd(),audio)

            features_predicted = self.hubert.get_transformation_layers(path)

            #Features predicted produces 12 layers of 20 ms frames of 768 features, we choose to use the 12th layer
            hubert_features[audio] = features_predicted[-1]

            if save:
                filepath = audio.replace(".wav","_features.npy")
                filenp = open(filepath,"bw+")
                features_numpy = hubert_features[audio].numpy()
                np.save(filenp,features_numpy)

        return hubert_features

    def fit(self,X,y):
        """ Trains the linear regression model utilizing the features provided by the user whose labels are mapped based
        on the tab-delimited txt.

            Args:
                X - A list of HuBERT frames utilized to fitting the Linear Regression model.
                labels - A list of tab-delimited text file that contains the channel,start and end time, and reduction values for the utterances
                         in the training data.
            Returns:
                None
            Exceptions:
                Invalid Arguments - Raised when X and y don't match in length, or when the file opened by y is in an incorrect format
        """
        if len(X) != len(y):
            raise ValueError("The lists of audio frame data and text files must match in length.")
        
        training_data = []
        training_labels = []
        for i in range(len(y)):
            filetxt = open(y[i])

            for line in filetxt:
                try:
                    _,channel,start,end,label = line.split()
                except: #Handling for erroneous format
                    incomplete += 1
                    continue

                label = int(label)

                #If Mono, the channel selected will be 0
                if channel == "Right":
                    x = 1
                else:
                    x = 0

                j = math.floor(float(start)*50) #Sets the index to the beginning of the labeled section
                while j <= math.floor(float(end)*50) and j < len(X[i][x]): #Iterates through each frame relating to the labeled section
                    training_data.append(X[i][x][j])
                    training_labels.append(label)
                    i += 1


        self.regression_model.fit(X,y)
        return

    def default_fit(self):
        """ Trains the linear regression model utilizing the data provided by ISG contained within the default_data directory.

            Args:
                None
            Returns:
                None
        """
                            
        default_X = [] 
        default_y = []
        default_files = ["EN_006","EN_007","EN_013","EN_033","EN_043",]
        for file in default_files:
            filenp = open(file+".npy","br")

            last_layer = None #Load the 12th last_layer, change the range for the wanted last_layer
            for i in range(12):
                last_layer = np.load(filenp)

            filetxt = open(file+".txt")

            for line in filetxt:
                try:
                    _,channel,start,end,label = line.split()
                except: #Handling for erroneous format
                    incomplete += 1
                    continue

                label = int(label)

                #If Mono, the channel selected will be 0
                if channel == "Right":
                    x = 1
                else:
                    x = 0

                i = math.floor(float(start)*50) #Sets the index to the beginning of the labeled section
                while i <= math.floor(float(end)*50) and i < len(last_layer[x]): #Iterates through each frame relating to the labeled section
                    default_X.append(last_layer[x][i])
                    default_y.append(label)
                    i += 1

        self.regression_model.fit(default_X,default_y)
        return

    def predict(self,audio_features):
        """ Predicts the reduction value per frame for a given list of HuBERT features corresponding to an audio.

            Args:
                audio_features - A list of HuBERT features extracted from an audio.
            Returns:
                predictions - A list of reduction values estimated for each 20 ms HuBERT frame.
        """
        predictions = []
        for frame in audio_features:
            predictions.append(self.regression_model.predict(frame)[0])
        return predictions

    def predict_utterances(self,audio_features,utterances):
        """ Predicts the reduction value per utterances based on the given list of HuBERT features. The predicted
        reduction value originates from the average of predicted frame value for each frame encompassed in the utterance region.
        The utterance region is determined by the start and end times found in the `utterances` file.

            Args:
                audio_features - A list of HuBERT features extracted from an audio.
                utterances - A path to tab-delimited text file that contains the channel, start and end times for the utterance to be predicted.
            Returns:
                predictions - A list of reduction values predicted for each utterance specified in the utterance file
        """
        predictions = []

        filetxt = open(utterances)
        for line in filetxt:

            try:
                _,channel,start,end,not_used_label = line.split() #TODO Fix the _ for the term "Reduction" and for the labels in data to be predicted
            except: #Handling for erroneous format
                incomplete += 1
                continue

            #If Mono, the channel selected will be 0
            if channel == "Right":
                x = 1
            else:
                x = 0

            X = [] #Region of HuBERT frames to predicted over
            i = math.floor(float(start)*50) #Sets the index to the beginning of the labeled section
            while i <= math.floor(float(end)*50) and i < len(audio_features[x]): #Iterates through each frame relating to the labeled section
                X.append(audio_features[x][i])
                i += 1
            
            frame_predictions = []
            for frame in X:
                frame_predictions.append(self.regression_model.predict(frame)[0])
            
            predictions.append(np.mean(frame_predictions))

        return predictions
    
    def calculate_overestimates(self,utterances,predicted):
        """ Returns the list of differences sorted by the highest overestimates accompanied with the timeframe for failure analysis.

            Args:
                utterances - A path to tab-delimited text file that contains the channel, start and end times for the utterance to be predicted.
                predicted - A list of predicted reduction values from the model
            Returns:
                overestimates - A list ordered of highest overestimates

        """
        labels = []
        filetxt = open(utterances)
        for line in filetxt:
            try:
                _0 ,_1, _2 ,_3, label = line.split() #TODO Fix the _ for the term "Reduction" and for the labels in data to be predicted
            except: #Handling for erroneous format
                incomplete += 1
                continue
            labels.append(int(label))

        labels = np.array(labels)
        predicted = np.array(predicted)

        overestimates = np.sort(labels-predicted)
        return overestimates
    
    def calculate_underestimates(self,utterances,predicted):
        """ Returns the list of differences sorted by the highest overestimates accompanied with the timeframe for failure analysis.

            Args:
                utterances - 
                predicted -
            Returns:
                overestimates -

        """
        labels = []
        filetxt = open(utterances)
        for line in filetxt:
            try:
                _0 ,_1, _2 ,_3, label = line.split() #TODO Fix the _ for the term "Reduction" and for the labels in data to be predicted
            except: #Handling for erroneous format
                incomplete += 1
                continue
            labels.append(int(label))

        labels = np.array(labels)
        predicted = np.array(predicted)

        underestimates = np.sort(labels-predicted)[::-1]
        return underestimates

