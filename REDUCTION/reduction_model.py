import os
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from .feature_extractor import FeatureExtractor


class Reduction:

    def __init__(self):
        self.regression_model = LinearRegression()
        self.hubert = FeatureExtractor()

    def extract(self,audios,save=False):
        """Extracts the 12th layer of HuBERT Base features for every audio in a given audio list.

        Args:
            audios - A list containing the path to wav files that will be processed using HuBERT.
            save - A flag that allows the features extracted to be saved in the same directory as the audio processed
                with the same name except having `features` appended.

        Returns:
            hubert_features - A list containing the 12th layer of hubert features for every audio processed. 
        
        """
        hubert_features = []
        for audio in audios:
            path = os.path.join(os.getcwd(),audio)

            features_predicted = self.hubert.get_transformation_layers(path)
            #Features predicted produces 12 layers of 20 ms frames of 768 features, we choose to use the 12th layer
            hubert_features.append(features_predicted[-1])

            if save:
                filepath = audio.replace(".wav","_features.npy")
                filenp = open(filepath,"bw+")
                features_numpy = features_predicted[-1].numpy()
                np.save(filenp,features_numpy)

        return hubert_features

    def fit(self,X,y):
        """ Trains the linear regression model utilizing the features provided by the user whose labels are mapped based
        on the tab-delimited txt.

            Args:
                X - A list of HuBERT frame lists utilized to fitting the Linear Regression model.
                labels - A list of tab-delimited text file that contains the channel,start and end time, and reduction values for the utterances
                         in the training data.
            Returns:
                None
            Exceptions:
                ValueError - Raised when X and y don't match in length
        """
        if len(X) != len(y):
            raise ValueError("The lists of audio frame data and text files must match in length.")
        
        training_data = []
        training_labels = []
        incomplete = 0
        for i in range(len(y)):
            filetxt = open(y[i])

            for line in filetxt:
                try:
                    channel,start,end,label = line.split()
                except: #Handling for erroneous format
                    incomplete += 1
                    continue

                label = int(label)

                #If Mono, the channel selected will be 0
                if channel == "Right":
                    track = 1
                else:
                    track = 0

                j = math.floor(float(start)*50) #Sets the index to the beginning of the labeled section
                while j <= math.floor(float(end)*50) and j < len(X[i][track]): #Iterates through each frame relating to the labeled section
                    training_data.append(X[i][track][j])
                    training_labels.append(label)
                    j += 1


        self.regression_model.fit(training_data,training_labels)
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
            filenp = open("./REDUCTION/default_data/"+file+".npy","br")

            last_layer = np.load(filenp) #Load the layer contained in the files which is the 12th layer.

            filetxt = open("./REDUCTION/default_data/"+file+".txt")

            for line in filetxt:
                try:
                    channel,start,end,label = line.split()
                except: 
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
        """ Predicts the reduction value per frame for a given list of HuBERT features corresponding to a singular audio.

            Args:
                audio_features - A list of frames of HuBERT features extracted from an audio. Since there only frames, it only supports one track.
            Returns:
                predictions - A list of reduction values estimated for each 20 ms HuBERT frame.
        """
        complete_predictions = []
        for i in range(len(audio_features)): #For every track in the audio features
            predictions = []
            for frame in audio_features[i]: #For every frame in an utterance region
                frame = np.array(frame).reshape(1,-1)
                predictions.append(self.regression_model.predict(frame)[0])
            complete_predictions.append(predictions)
        return complete_predictions

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

        # TODO Create a copy of the txt with the predicted labels
        filetxt = open(utterances)
        for line in filetxt:

            try:
                channel,start,end = line.split()
            except: #Handling for erroneous format
                print(f"Could not handle the following line during prediction:\n {line} The line will be skipped.\n")
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
                frame = np.array(frame).reshape(1,-1)
                frame_predictions.append(self.regression_model.predict(frame)[0])
            
            predictions.append(np.mean(frame_predictions))

        return predictions
    
    def calculate_overestimates(self,predicted,utterances):
        """ Returns the list of differences sorted by the highest overestimates accompanied with the timeframe for failure analysis.

            Args:
                utterances - A path to tab-delimited text file that contains the channel, start and end times, and labels for the utterance to be predicted.
                predicted - A list of predicted reduction values from the model
            Returns:
                overestimates - A list ordered of highest overestimates between the actual and predicted reduction values.

        """
        channels = []
        starts = []
        ends = []
        labels = []
        filetxt = open(utterances)
        valid_index = []
        i = 0
        for line in filetxt:
            try:
                channel , start, end , label = line.split()
            except: #Handling for erroneous format
                print(f"Could not handle the following line while calculating overestimates:\n {line} The line will be skipped.\n")
                continue

            valid_index.append(i)
            i += 1
            channels.append(channel)
            starts.append(start)
            ends.append(end)
            labels.append(int(label))

        labels = np.array(labels)
        predicted = np.array(predicted)

        overestimates_index = np.argsort(labels[valid_index]-predicted[valid_index])
        overestimates = labels[valid_index]-predicted[valid_index]

        #Creates a sorted array of formatted strings based on the overestimates containing the time and utterance for the channel
        formatted_overestimates = [f"Difference: {overestimates[i]}, Channel: {channels[i]}, Start Time: {starts[i]}, End Time: {ends[i]}, Actual Reduction: {labels[i]}, Predicted Reduction: {predicted[i]}" for i in overestimates_index]

        return formatted_overestimates
    
    def calculate_underestimates(self,predicted,utterances):
        """ Returns the list of differences sorted by the highest overestimates accompanied with the timeframe for failure analysis.

            Args:
                utterances - A path to tab-delimited text file that contains the channel, start and end times, and labels for the utterance to be predicted.
                predicted - A list of predicted reduction values from the model
            Returns:
                underestimates - A list ordered of highest underestimates between the actual and predicted reduction values.

        """
        channels = []
        starts = []
        ends = []
        labels = []
        filetxt = open(utterances)
        valid_index = []
        i = 0
        for line in filetxt:
            try:
                channel , start, end , label = line.split()
            except: #Handling for erroneous format
                print(f"Could not handle the following line while calculating underestimates:\n {line} The line will be skipped.\n")
                continue

            valid_index.append(i)
            i += 1
            channels.append(channel)
            starts.append(start)
            ends.append(end)
            labels.append(int(label))

        labels = np.array(labels)
        predicted = np.array(predicted)

        underestimates_index = np.argsort(labels[valid_index]-predicted[valid_index])[::-1]
        underestimates = labels[valid_index]-predicted[valid_index]

        #Creates a sorted array of formatted strings based on the underestimates containing the time and utterance for the channel
        formatted_underestimates = [f"Difference: {underestimates[i]}, Channel: {channels[i]}, Start Time: {starts[i]}, End Time: {ends[i]}, Actual Reduction: {labels[i]}, Predicted Reduction: {predicted[i]}" for i in underestimates_index]

        return formatted_underestimates

