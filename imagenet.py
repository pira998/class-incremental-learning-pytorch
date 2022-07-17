import os
from continuum.datasets import ImageFolderDataset

class ImageNet1000(ImageFolderDataset):
    def __init__(
        self,
        data_path: str,
        train:bool = True,
        download:bool = False
    ):
        super().__init__(data_path, train, download)
    
    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, 'train')      
        else:
            self.data_path = os.path.join(self.data_path, 'val')
        return super().get_data()




