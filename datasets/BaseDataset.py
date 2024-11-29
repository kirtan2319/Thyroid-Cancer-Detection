import torch
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, path_to_images, sens_name, sens_classes, transform):
        # Initialize the base class
        super(BaseDataset, self).__init__()
        
        self.dataframe = dataframe        
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.path_to_images = path_to_images
        self.sens_name = sens_name
        self.sens_classes = sens_classes
        
        self.A = None
        self.Y = None
        self.AY_proportion = None
        
    def get_AY_proportions(self):
        # Calculate AY proportions if not already computed
        if self.AY_proportion:
            return self.AY_proportion
        
        A_num_class = 2
        Y_num_class = 2
        A_label = self.A
        Y_label = self.Y
        
        A = self.A.tolist()
        Y = self.Y.tolist()
        ttl = len(A)
        
        # Calculate group counts for AY proportions
        len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
        len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
        len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
        len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

        assert (
            len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1
        ) == ttl, "Problem computing train set AY proportion."
        
        A0Y0 = len_A0Y0 / ttl
        A0Y1 = len_A0Y1 / ttl
        A1Y0 = len_A1Y0 / ttl
        A1Y1 = len_A1Y1 / ttl
        
        self.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]
        
        return self.AY_proportion
    
    def get_A_proportions(self):
        # Get proportions of the sensitive attribute A
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def get_Y_proportions(self):
        # Get proportions of the label Y
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def set_A(self, sens_name):
        # Set sensitive attribute A based on sens_name
        if sens_name == 'Sex':
            A = np.asarray(self.dataframe['Sex'].values != 'M').astype('float')
        elif sens_name == 'Age':
            A = np.asarray(self.dataframe['Age_binary'].values.astype('int') == 1).astype('float')
        elif sens_name == 'Race':
            A = np.asarray(self.dataframe['Race'].values == 'White').astype('float')
        elif sens_name == 'skin_type':
            A = np.asarray(self.dataframe['skin_binary'].values != 0).astype('float')
        elif sens_name == 'Insurance':
            A = np.asarray(self.dataframe['Insurance_binary'].values != 0).astype('float')
        else:
            raise ValueError(f"Does not contain {sens_name}")
        return A

    def get_weights(self, resample_which):
        # Get sample weights for resampling
        sens_attr, group_num = self.group_counts(resample_which)
        group_weights = [1/x.item() for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights
    
    def group_counts(self, resample_which='group'):
        # Count group occurrences for resampling
        if resample_which == 'group' or resample_which == 'balanced':
            if self.sens_name == 'Sex':
                mapping = {'M': 0, 'F': 1}
                groups = self.dataframe['Sex'].values
                group_array = [*map(mapping.get, groups)]
                
            elif self.sens_name == 'Age':
                if self.sens_classes == 2:
                    groups = self.dataframe['Age_binary'].values
                elif self.sens_classes == 5:
                    groups = self.dataframe['Age_multi'].values
                elif self.sens_classes == 4:
                    groups = self.dataframe['Age_multi4'].values.astype('int')
                group_array = groups.tolist()
                
            elif self.sens_name == 'Race':
                mapping = {'White': 0, 'non-White': 1}
                groups = self.dataframe['Race'].values
                group_array = [*map(mapping.get, groups)]
                
            elif self.sens_name == 'skin_type':
                if self.sens_classes == 2:
                    groups = self.dataframe['skin_binary'].values
                elif self.sens_classes == 6:
                    groups = self.dataframe['skin_type'].values
                group_array = groups.tolist()
                
            elif self.sens_name == 'Insurance':
                if self.sens_classes == 2:
                    groups = self.dataframe['Insurance_binary'].values
                elif self.sens_classes == 5:
                    groups = self.dataframe['Insurance'].values
                group_array = groups.tolist()
            else:
                raise ValueError("Sensitive attribute not defined in BaseDataset")
            
            if resample_which == 'balanced':
                # Adjust for class balancing
                labels = self.Y.tolist()
                num_labels = len(set(labels))
                num_groups = len(set(group_array))
                
                group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()
                
        elif resample_which == 'class':
            group_array = self.Y.tolist()
            num_labels = len(set(group_array))
        
        self._group_array = torch.LongTensor(group_array)
        
        # Count occurrences of groups
        if resample_which == 'group':
            self._group_counts = (torch.arange(self.sens_classes).unsqueeze(1) == self._group_array).sum(1).float()
        elif resample_which == 'balanced':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1) == self._group_array).sum(1).float()
        elif resample_which == 'class':
            self._group_counts = (torch.arange(num_labels).unsqueeze(1) == self._group_array).sum(1).float()
        
        return group_array, self._group_counts
    
    def __len__(self):
        # Return the size of the dataset
        return self.dataset_size
    
    def get_labels(self): 
        # Return the sensitive attribute labels
        if self.sens_classes == 2:
            return self.A
        elif self.sens_classes == 5:
            return self.dataframe['Age_multi'].values.tolist()
        elif self.sens_classes == 4:
            return self.dataframe['Age_multi4'].values.tolist()

    def get_sensitive(self, sens_name, sens_classes, item):
        # Get the sensitive attribute for a specific item
        if sens_name == 'Sex':
            sensitive = 0 if item['Sex'] == 'M' else 1
        elif sens_name == 'Age':
            if sens_classes == 2:
                sensitive = int(item['Age_binary'])
            elif sens_classes == 5:
                sensitive = int(item['Age_multi'])
            elif sens_classes == 4:
                sensitive = int(item['Age_multi4'])
        elif sens_name == 'Race':
            sensitive = 0 if item['Race'] == 'White' else 1
        elif sens_name == 'skin_type':
            sensitive = int(item['skin_binary']) if sens_classes == 2 else int(item['skin_type'])
        elif sens_name == 'Insurance':
            sensitive = int(item['Insurance_binary']) if sens_classes == 2 else int(item['Insurance'])
        else:
            raise ValueError(f"Unknown sensitive attribute: {sens_name}")
        
        return sensitive
