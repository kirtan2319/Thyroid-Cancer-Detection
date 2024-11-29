import torch
import pickle
import numpy as np
from PIL import Image
from datasets.BaseDataset import BaseDataset


class Thy(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        # Initialize the parent class
        super(Thy, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)
        
        # Load the pickle data
        with open(path_to_pickles, 'rb') as f:
            self.tol_images = pickle.load(f)
        
        # Set up sensitive attribute and labels
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['binaryLabel'].values) > 0).astype('float')
        self.AY_proportion = None

    def __getitem__(self, idx):
        # Retrieve item information from dataframe
        item = self.dataframe.iloc[idx]

        # Load and transform the image
        img = Image.fromarray(self.tol_images[idx])
        #print(f"Image at index {idx}: Type = {type(self.tol_images[idx])}, Shape = {self.tol_images[idx].shape}")

        # Apply the image transformation
        img = self.transform(img)

        # Retrieve the label and convert to a tensor
        label = torch.FloatTensor([int(item['binaryLabel'])])
        #print(f"Label at index {idx}: {label}")

        # Get sensitive attribute information
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
        
        # Return the image, label, sensitive attribute, and index
        return img, label, sensitive, idx
