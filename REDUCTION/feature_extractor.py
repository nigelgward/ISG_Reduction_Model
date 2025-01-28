import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


class FeatureExtractor:
    """
    Uses pytorch to extract transformation layers from an audio file.
    Utilizes either the HuBERT Large, Wav2Vec2.0 Large, or WavLM Large models
    bundle: String : 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected
    """
    def __init__(self, bundle='hubert_b'):
        torch.random.manual_seed(0)  # Sets the same random weights everytime the model is run.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA uses GPUs for computations.
        self.bundle = self.get_bundle(bundle)
        self.model = self.bundle.get_model().to(self.device)
        self.print_info()

    def get_bundle(self, bundle):
        if bundle == "hubert_l": return torchaudio.pipelines.HUBERT_LARGE
        if bundle == "wav2vec_l": return torchaudio.pipelines.WAV2VEC2_LARGE
        if bundle == "wavlm_l": return torchaudio.pipelines.WAVLM_LARGE
        if bundle == "hubert_b": return torchaudio.pipelines.HUBERT_BASE
        print(f"bundle name {bundle} not recognized: 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected.")
        sys.exit()

    def print_info(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch audio version: {torchaudio.__version__}")
        print(f"device: {self.device}")
        print(f"Sample Rate: {self.bundle.sample_rate}")
        print(f"model class: {self.model.__class__}")

    def get_transformation_layers(self, path, plot_layers=False):
        """
        Passes an audio file to a self-supervised machine learning model.
        :param path: path of the audio file : String
        :param plot_layers: Will visualize the transformation layers if true.
        :return: A list of 24 3d tensors representing the 24 transformation layers
        Tensor Dimensions:
        1st = number of audio files processed at once
        2nd = number of frames per audio file (One frame for every 10ms)
        3rd = number of features extracted per frame (1024)
        """
        if not os.path.exists(path):
            print(f"ERROR: {path} is not a valid file path")
            return
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(self.device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        with torch.inference_mode():  # Disables gradient computation and back propagation.
            features, _ = self.model.extract_features(waveform)
        if plot_layers: self.plot_layers(features)
        return features
