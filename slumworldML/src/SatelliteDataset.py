import sys
import os
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from typing import Optional, List, Dict, Any
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import  LightningDataModule
from packaging import version

from pytorch_lightning.trainer.supporters import CombinedLoader
# from lightning.pytorch.utilities import CombinedLoader


try:
    from slumworldML.src.transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                                    INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)
    from slumworldML.src.custom_transformations import RandomRotateZoomCropTensor, LabelNoise
except Exception as Error:
    try:
        from src.custom_transformations import RandomRotateZoomCropTensor, LabelNoise
        from src.transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                            INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)
    except Exception as Error:
        from custom_transformations import  RandomRotateZoomCropTensor, LabelNoise
        from transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                        INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)

import pdb


class BaseCNNDataset(torch.utils.data.Dataset):

    @staticmethod
    def create_save_name(filename, y_loc, x_loc):
        coor = filename.split(os.sep)[-1]
        coor = os.path.splitext(coor)[0]
        coor = coor.split("_")
        y_loc_tile = int(coor[0]) + y_loc
        x_loc_tile = int(coor[1]) + x_loc

        return str(y_loc_tile) + "_" + str(x_loc_tile) + ".png"

    def _get_x_y_filename(self, index):
        '''Generates one sample of data when the crop_split_resize transform is applied hence
        there is no need to split the tile into subtiles'''
        # Select sample
        ## SOS CHANGE THIS
        if self.df_other is not None:
            if np.random.choice([0,1,2]) > 0:
                # sample from slum tiles with 66% probability
                index = index % (self.orsize)
                # index = index % (self.orsize-1)
                filename_x = self.df.iloc[index]["x_location"]
                filename_y = self.df.iloc[index]["y_location"]
            else:
                # sample from all other tiles
                index = np.random.choice(len(self.df_other))
                filename_x = self.df_other.iloc[index]["x_location"]
                filename_y = self.df_other.iloc[index]["y_location"]
        else:
            index = index % (self.orsize-1)
            filename_x = self.df.iloc[index]["x_location"]
            filename_y = self.df.iloc[index]["y_location"]

        if self.from_disc:
            x = read_image(filename_x)
            y = read_image(filename_y)
        else:
            x = self.read_image_from_disc(filename_x, islabel=False)
            y = self.read_image_from_disc(filename_y, islabel=True)

        if self.transform is not False:
                x, y = self.transform(x, y)
        return x, y, filename_x

    def _get_x_filename(self, index):
        '''Generates one sample of data without labels when the crop_split_resize transform is applied hence
        there is no need to split the tile into subtiles'''
        # Select sample
        index = index % (self.orsize-1)
        filename_x = self.df.iloc[index]["x_location"]
        if self.from_disc:
            x = read_image(filename_x)
        else:
            x = self.read_image_from_disc(filename_x, islabel=False)

        if self.transform is not False:
            x = self.transform(x, None)
        return x, filename_x

    def read_image_from_disc(self, filename, islabel=False):
        if islabel:
            out = copy.deepcopy(self.y_dict[filename])
        else:
            out = copy.deepcopy(self.x_dict[filename])
        return out

    def _load_to_ram(self, df):
        '''load the whole dataset to RAM'''
        # process x 
        for file in df.x_location:
            self.x_dict[file] = read_image(file)
        # process y
        try:
            for file in df.y_location:
                self.y_dict[file] = read_image(file)
        except Exception as Error:
            pass


class TrainDataSet(BaseCNNDataset):

    def __init__(self, balancing_df, complement_df=None, transform=False, tile_size=None, size_mutliplier=10, in_ram_dataset=False):
        '''
                balancing_df:           dataframe, contains references to mostly minority class data
                complement_df:          dataframe, contains references to all other data
                                        returned values will be split 50-50 between the two dataframes
                transform:              function, composed transforms to use
                tile_size:              int, the dataset tile size
                size_mutliplier:        int, how many times to loop through the minority class (augmented) tiles data per epoch
                in_ram_dataset:    boolean, if set to true the whole data will be loaded into memory [default: false]
                '''
        super().__init__()
        self.df = balancing_df
        self.df_other = complement_df
        self.transform = transform
        self.orsize = len(self.df)
        self.from_disc = not in_ram_dataset 
        self.x_dict = {}
        self.y_dict = {}
        if in_ram_dataset:
            self._load_to_ram(self.df)
            if self.df_other is not None:
                self._load_to_ram(self.df_other)
        transform_types = [str(type(t)) for t in transform.joint_transforms]
        if str(type(RandomRotateZoomCropTensor())) in transform_types:
            # we do not need to split the tile it will be handled inside the transforms
            self.SPLIT_TILE = False
            self.size_multiplier = size_mutliplier
        else:
            self.SPLIT_TILE = True
            if tile_size is None:
                print("Tile size not provided. Assuming an input tile_size of 512.")
                original_size = 512
            else:
                original_size = tile_size
            self.subtile_size = int(original_size/2)
            self.tiles_split_multiplier = int(original_size/self.subtile_size)**2
            self.size_multiplier = size_mutliplier if size_mutliplier >=4 else self.tiles_split_multiplier
            # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
            coords = [i*self.subtile_size for i in range(int(original_size/self.subtile_size))]
            self.top_left_point = []
            for height in coords:
                for width in coords:
                        self.top_left_point.append([height, width])

    def __len__(self):
        'Denotes the total number of samples'
        ## SOS CHANGE THIS
        return len(self.df) * self.size_multiplier 

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y, filename_x = self._get_x_y_filename(index)
        if not self.SPLIT_TILE:
            '''Generates one sample of data when the crop_split_resize transform is applied hence
                there is no need to split the tile into subtiles'''
            save_name = filename_x.split(os.sep)[-1]
            return x, y, save_name
        else:
            # select one of the possible subtiles
            tile_number = np.random.choice(list(range(self.tiles_split_multiplier)))
            # get the height and width values for the current sub-tile
            y_loc = self.top_left_point[tile_number][0]
            x_loc = self.top_left_point[tile_number][1]
            x_ = x[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size ]
            y_ = y[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size ]
            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            return x_, y_, save_name


class ValidationDataset(BaseCNNDataset):

    def __init__(self, validation_df, transform=False, tile_size=None, 
                 split_tiles=True, in_ram_dataset=False, mode='eval'):
        super().__init__()
        self.df = validation_df
        self.transform = transform
        self.from_disc = True
        self.eval_mode = (mode == 'eval') or (mode == 'evaluation')
        self.y_dict = None
        self.x_dict = None
        if tile_size is None:
                print("Tile size not provided. Assuming an input tile_size of 512.")
                original_size = 512
        else:
                original_size = tile_size
        self.split_tiles = split_tiles
        if split_tiles:
            self.subtile_size = int(original_size/2)
            self.tiles_split_multiplier = int(original_size/self.subtile_size )**2
            # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
            coords = [i*self.subtile_size for i in range(int(original_size/self.subtile_size ))]
            self.top_left_point = []
            for height in coords:
                    for width in coords:
                        self.top_left_point.append([height, width])
        else: 
            self.tiles_split_multiplier = 1

    def __len__(self):
        '''Denotes the total number of samples'''        
        return len(self.df)*self.tiles_split_multiplier

    def __getitem__(self, index):
        '''Crops a large tile into 4 x (tile_size/2 x tile_size/2) subtiles for compatibility
        with the training setup (where each tile is sampled at size tile_size/2).'''
        df_index = int(index/self.tiles_split_multiplier)   # int acts as floor operation
        if self.split_tiles:
            tile_number = index%self.tiles_split_multiplier     # remainder from division to choose tile
            # get the height and width values for the current sub-tile
            y_loc = self.top_left_point[tile_number][0]
            x_loc = self.top_left_point[tile_number][1]

            'Generates one sample of data'
            # Select sample
            filename_x = self.df.iloc[df_index]["x_location"]
            x = read_image(filename_x)

            if self.eval_mode: 
                filename_y = self.df.iloc[df_index]["y_location"]
                y = read_image(filename_y)
                if self.transform is not False:
                    x, y = self.transform(x, y)
            else:
                if self.transform is not False:
                    x = self.transform(x)

            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            x_ = x[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size ]
            
            if self.eval_mode: 
                y_ = y[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size ]
                return x_, y_, save_name
            else:
                return x_, save_name
        else:
            # Select sample
            filename_x = self.df.iloc[df_index]["x_location"]
             # Load image
            x = read_image(filename_x)
            save_name = filename_x.split(os.sep)[-1]

            if self.eval_mode: 
                filename_y = self.df.iloc[df_index]["y_location"]
                y = read_image(filename_y)
                if self.transform is not False:
                    x, y = self.transform(x, y)
            else:
                if self.transform is not False:
                    x = self.transform(x)

            if self.eval_mode: 
                return x, y, save_name
            else:
                return x, save_name

class InferenceDataset(torch.utils.data.Dataset):
    '''Used when one needs to perform inference directly on a folder with (tiled) images, i.e. without a dataset.csv'''
    def __init__(self, tiled_folder_path, transform=False, tile_size=None, in_ram_dataset=False):
        super().__init__()
        # get all items from the folder
        # all .png files
        self.tile_filenames = [f for f in os.listdir(tiled_folder_path) if f.split('.')[-1] in ['png', 'jpg', 'jpeg']]
        self.tiled_folder_path = tiled_folder_path
        self.transform = transform
        self.from_disc = True
        self.y_dict = None
        self.x_dict = None
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.tile_filenames)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        filename_x = self.tile_filenames[index]

        # Load each image
        # x = read_image(os.path.join(self.tiled_folder_path, filename_x)).float()
        x = read_image(os.path.join(self.tiled_folder_path, filename_x))

        if self.transform is not False:
                x = self.transform(x)

        return x, filename_x

class AdaptationDataset(BaseCNNDataset):

    def __init__(self, adaptation_df, transform=False, tile_size=None, size_mutliplier=4, in_ram_dataset=False):
        super().__init__()
        self.df = adaptation_df
        self.transform = transform
        self.orsize = len(self.df)
        self.from_disc = not in_ram_dataset 
        self.x_dict = {}
        self.y_dict = {}
        if in_ram_dataset:
            self._load_to_ram(self.df)
        self.x_dict = None
        transform_types = [str(type(t)) for t in transform.joint_transforms]
        if str(type(RandomRotateZoomCropTensor())) in transform_types:
            # we do not need to split the tile it will be handled inside the transforms
            self.SPLIT_TILE = False
            self.tiles_split_multiplier = 1
            self.size_multiplier = size_mutliplier
        else:
            self.SPLIT_TILE = True
            if tile_size is None:
                print("Tile size not provided. Assuming an input tile_size of 512.")
                original_size = 512
            else:
                original_size = tile_size
            self.subtile_size = int(original_size/2)
            self.tiles_split_multiplier = int(original_size/self.subtile_size)**2
            self.size_multiplier = size_mutliplier if size_mutliplier >=4 else self.tiles_split_multiplier
            # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
            coords = [i*self.subtile_size for i in range(int(original_size/self.subtile_size))]
            self.top_left_point = []
            for height in coords:
                for width in coords:
                        self.top_left_point.append([height, width])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.df)*self.size_multiplier

    def __getitem__(self, index):
        'Generates one sample of data'
        x, filename_x = self._get_x_filename(index)
        if not self.SPLIT_TILE:
            '''Generates one sample of data when the crop_split_resize transform is applied hence
                there is no need to split the tile into subtiles'''
            save_name = filename_x.split(os.sep)[-1]
            return x, save_name
        else:
            # select one of the possible subtiles
            tile_number = np.random.choice(list(range(self.tiles_split_multiplier)))
            # get the height and width values for the current sub-tile
            y_loc = self.top_left_point[tile_number][0]
            x_loc = self.top_left_point[tile_number][1]
            x_ = x[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size ]
            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            return x_, save_name



class SatelliteDataModule(LightningDataModule):
    
    def __init__(self, dataset_file='/home/minas/slumworld/data/tiled/MS_MUL_75_Briana/dataset.csv', adaption_task_dataset_file=None,
                 required_slum_coverage=0.00001, size_mutliplier=8, norm_file=None, test_datapath=None, train_batch_size=4, test_batch_size=10, 
                 num_workers=4, shuffle=True, tile_size=None, foldID=-1, image_type='mul', overfit_mode=False, ssp=False, mask_sample_fraction=0.5,
                 label_noise=False, in_ram_dataset=False, training_transformations=None, validation_transformations=None):
        super().__init__()
        self.dataset_file = dataset_file
        self.adaption_dataset_file = adaption_task_dataset_file
        self.with_adaption_task = adaption_task_dataset_file is not None
        self.test_datapath = test_datapath
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.slum_coverage = required_slum_coverage
        self.size_mutliplier = size_mutliplier
        self.tile_size = tile_size
        self.foldID = foldID
        self.overfit_mode = overfit_mode
        self.PAN_IMAGE = image_type == 'pan'
        self.image_type = image_type
        self.MASK_SAMPLE_FRACTION = mask_sample_fraction #for diversity
        self.label_noise = label_noise
        self.in_ram_dataset = in_ram_dataset
        self.training_transformations = training_transformations
        self.validation_transformations = validation_transformations
        if ssp is not None:
            self.mode = ssp
        else:
            self.mode = 'train'
        self.df = pd.read_csv(self.dataset_file)
        if self.with_adaption_task:
            self.adf = pd.read_csv(self.adaption_dataset_file)
        else:
            self.adf = None
        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = data["mean"]
            self.std = data["std"]
            if len(self.mean) == 1: # PAN Image
                self.PAN_IMAGE = True
        else:
            print("Normalization File not Provided. Predictions will not be performed. Exiting...")
            sys.exit(1)
        self.setup(stage=None)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # create the dataset do the splits and everything here
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.foldID == -1:
            training_column = "dataset_part"
        elif isinstance(self.foldID, int):
            training_column = f"Fold_{self.foldID}"
        self.df.fillna(0.0, inplace=True)
        self.df_train = self.df[self.df[training_column] == "Train"]
        self.df_mask_sample = self.df[self.df[training_column] == "Mask"].sample(frac=self.MASK_SAMPLE_FRACTION)
        # we want data from the unmasked part of the satellite image to be more common than data from the masked areas
        # therefore, as a hack, we oversample them manually
        self.df_train_other = pd.concat([self.df_train[self.df_train["average_slum_coverage"] < self.slum_coverage], 
                                         self.df_train[self.df_train["average_slum_coverage"] < self.slum_coverage],
                                         self.df_mask_sample] ).reset_index(drop=True)
        self.df_train = self.df_train[self.df_train["average_slum_coverage"] >= self.slum_coverage]
        self.df_validation = self.df[self.df[training_column] == "Validation"]
        self.df_test = self.df[self.df[training_column] == "Test"]
        if self.with_adaption_task:
            self.adf.fillna(0.0, inplace=True)
            self.adf_train = self.adf[self.adf[training_column].isin(['Train', 'Mask'])]
            if len(self.adf[training_column] == 'Validation') > 0:
                self.adf_validation = self.adf[self.adf[training_column] == 'Validation']
            else:
                print("Warning!Adaptation dataset has no validation split.\n Will use the training split for validation")
                self.adf_validation = self.adf_train
        if self.PAN_IMAGE:
            if self.mode == "ssp":
                TRAIN_TRANSFORMS = SSP_TRANSFORMS_PAN
                INFER_TRANSFORMS = SSP_INFERENCE_TRANSFORMS
            elif self.mode == "assp":
                TRAIN_TRANSFORMS = ASSP_TRANSFORMS_PAN
                INFER_TRANSFORMS = ASSP_INFERENCE_TRANSFORMS
            else:
                TRAIN_TRANSFORMS = TRAINING_TRANSFORMS_PAN
                INFER_TRANSFORMS = INFERENCE_TRANSFORMS
        else:
            if self.mode == "ssp":
                TRAIN_TRANSFORMS = SSP_TRANSFORMS
                INFER_TRANSFORMS = SSP_INFERENCE_TRANSFORMS
            elif self.mode == "assp":
                TRAIN_TRANSFORMS = ASSP_TRANSFORMS
                INFER_TRANSFORMS = ASSP_INFERENCE_TRANSFORMS
            else:
                TRAIN_TRANSFORMS = TRAINING_TRANSFORMS
                INFER_TRANSFORMS = INFERENCE_TRANSFORMS
        if self.training_transformations is not None:
            TRAIN_TRANSFORMS = self.training_transformations
        if self.validation_transformations is not None:
            INFER_TRANSFORMS = self.validation_transformations
        if self.overfit_mode:
            # in overfit mode you do not need any transformations since you just want to see if you can overfit a few traning batches 
            TRAIN_TRANSFORMS = INFERENCE_TRANSFORMS
            complement_df = None
            self.shuffle = False
        else:
            complement_df = self.df_train_other
        if self.label_noise:
            TRAIN_TRANSFORMS['joint_transforms'].append(LabelNoise(noise_level=0.1))
            INFER_TRANSFORMS['joint_transforms'].append(LabelNoise(noise_level=0.1))
        train_transforms = create_transform(TRAIN_TRANSFORMS, mean=self.mean, std=self.std, input_size=self.tile_size//2)
        infer_transforms = create_transform(INFER_TRANSFORMS, mean=self.mean, std=self.std, input_size=self.tile_size//2)
        if (stage == 'train') or (stage is None):  
            self.train_dataset = TrainDataSet(balancing_df=self.df_train, complement_df=complement_df, transform=train_transforms,
                                              tile_size=self.tile_size,  size_mutliplier=self.size_mutliplier, in_ram_dataset=self.in_ram_dataset)
            self.validation_dataset = ValidationDataset(validation_df=self.df_validation, transform=infer_transforms, 
                                                        tile_size=self.tile_size)
            self.test_dataset = ValidationDataset(validation_df=self.df_test, transform=infer_transforms, 
                                                        tile_size=self.tile_size)
            if self.with_adaption_task:
                self.adaptation_train_dataset = AdaptationDataset(adaptation_df=self.adf_train, transform=copy.deepcopy(train_transforms), 
                                                                  tile_size=self.tile_size, in_ram_dataset=self.in_ram_dataset)
                self.adaptation_validation_dataset = AdaptationDataset(adaptation_df=self.adf_validation, transform=infer_transforms, 
                                                                  tile_size=self.tile_size)
            self.tiles_per_epoch = len(self.train_dataset)
            # === Debugging ===
            print(f"DEBUG_SAT_DM: In setup (stage={stage})")
            print(f"DEBUG_SAT_DM: len(self.df) = {len(self.df)}")
            print(f"DEBUG_SAT_DM: len(self.df_train) = {len(self.df_train)}")
            print(f"DEBUG_SAT_DM: len(self.df_validation) = {len(self.df_validation)}")
            print(f"DEBUG_SAT_DM: len(self.df_test) = {len(self.df_test)}")
            print(f"DEBUG_SAT_DM: len(self.train_dataset) (tiles_per_epoch) = {len(self.train_dataset)}")
            print(f"DEBUG_SAT_DM: len(self.validation_dataset) = {len(self.validation_dataset)}")
            if self.with_adaption_task:
                print(f"DEBUG_SAT_DM: len(self.adaptation_train_dataset) = {len(self.adaptation_train_dataset)}")
                print(f"DEBUG_SAT_DM: len(self.adaptation_validation_dataset) = {len(self.adaptation_validation_dataset)}")
            # ======================================
        elif stage == "test":
            self.test_dataset = InferenceDataset(tiled_folder_path=self.test_datapath, transform=infer_transforms, tile_size=self.tile_size)
            # === Debugging ===
            print(f"DEBUG_SAT_DM: In setup (stage={stage})")
            print(f"DEBUG_SAT_DM: len(self.test_dataset) = {len(self.test_dataset)}")
            # ==========================================
        else:
            print("Stage not understood. Must be one of \'train\', \'test', or, \'None\'.")

    def train_dataloader(self):
        if self.with_adaption_task:
            train_dataloader = DataLoader(self.train_dataset, batch_size = self.train_batch_size//2, num_workers=self.num_workers//2,
                                        persistent_workers=True, shuffle=self.shuffle, pin_memory=True, prefetch_factor=2)
            adaptation_dataloader = DataLoader(self.adaptation_train_dataset, batch_size = self.train_batch_size//2, num_workers=self.num_workers//2,
                                               persistent_workers=True, shuffle=self.shuffle, pin_memory=True, prefetch_factor=2)
            dataloaders_dict = {'slum_prediction': train_dataloader, 'domain_prediction': adaptation_dataloader} 
        else:
            train_dataloader = DataLoader(self.train_dataset, batch_size = self.train_batch_size, num_workers=self.num_workers//2,
                                        persistent_workers=True, shuffle=self.shuffle, pin_memory=True, prefetch_factor=2)
            dataloaders_dict = {'slum_prediction': train_dataloader} 
        return CombinedLoader(dataloaders_dict, mode='min_size')

    def val_dataloader(self):
        # === Check validation set length when this func called - debugging ===
        print(f"DEBUG_SAT_DM: val_dataloader called. Current epoch: {self.trainer.current_epoch if hasattr(self, 'trainer') else 'N/A'}")
        print(f"DEBUG_SAT_DM: val_dataloader - len(self.validation_dataset) at call = {len(self.validation_dataset)}")
        # =============================================================

        if self.with_adaption_task:
            validation_dataloader = DataLoader(self.validation_dataset, batch_size = self.train_batch_size//2, num_workers=self.num_workers//2, 
                                            persistent_workers=True, shuffle=False, pin_memory=True)
            adaptation_dataloader = DataLoader(self.adaptation_validation_dataset, batch_size = self.train_batch_size//2, num_workers=self.num_workers//2,
                                        persistent_workers=True, shuffle=False, pin_memory=True, prefetch_factor=2)
            Loaders_dict = {'slum_prediction': validation_dataloader, 'domain_prediction': adaptation_dataloader}
        else:
            validation_dataloader = DataLoader(self.validation_dataset, batch_size = self.train_batch_size, num_workers=self.num_workers//2, 
                                            persistent_workers=True, shuffle=False, pin_memory=True)
            Loaders_dict = {'slum_prediction': validation_dataloader}
        # === Print of meta data of dataloader - debugging ===
        print(f"DEBUG_SAT_DM: val_dataloader - validation_dataloader batch_size={validation_dataloader.batch_size}, num_workers={validation_dataloader.num_workers}, persistent_workers={validation_dataloader.persistent_workers}")
        # ====================================================================

        return [CombinedLoader(Loaders_dict, mode='min_size')]

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size = self.test_batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        dataloaders_dict = {'slum_prediction': test_dataloader} 
        return CombinedLoader(dataloaders_dict, mode='min_size')

    def predict_dataloader(self):
        evaluation_dataloader = DataLoader(self.test_dataset, batch_size = self.test_batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        dataloaders_dict = {'slum_prediction': evaluation_dataloader} 
        return CombinedLoader(dataloaders_dict, mode='min_size')

class InferenceDataLoader:

    def __init__(self, datapath='/home/minas/slumworld/data/tiled/MS_MUL_75_Briana/tiled_input/dataset.csv', norm_file=None,
                 batch_size=25, num_workers=4, tile_size=None, shuffle=False):
        super().__init__()
        if tile_size is None:
            df = pd.read_csv(datapath)
            tile_size = io.imread(df.iloc[0,0]).shape[1]
        self.datapath =datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tile_size = tile_size
        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = data["mean"]
            self.std = data["std"]
        else:
            print("Normalization File not Provided. Will not normalize. Predictions prone to error.")
            self.mean = np.array([0.])
            self.std = np.array([1.])
        self.setup()

    def setup(self):
        infer_transforms = create_transform(INFERENCE_TRANSFORMS, mean=self.mean, std=self.std, input_size=self.tile_size//2)
        ## TODO
        self.test_dataset = InferenceDataset(tiled_folder_path=self.datapath, 
                                                transform=infer_transforms)

    def get_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size = self.batch_size, 
                                     num_workers=self.num_workers, shuffle=self.shuffle)
        return test_dataloader


class TestingDataLoader:
    ''' DataLoader to be used for Evaluation/Testing
    Loads all tiles from a tiled folder, normalizes and presents them to the model.
    ARGS:
        dataset_file:   str, full path to the dataset.csv file [default: None]
        norm_file       str, full path to the normalization file used during training [default: None]
        batch_size      int, batch size [default: 10]
        num_workers     int, number of workers to use for loading and preprocessing images [default: 2]
        tile_size:      int, [default: 512]
        shuffle:        boolean, shuffle data [default: False]
        foldID:         int, for k-fold CV, -1 for standard training [default: -1]
        split_tiles     boolean, split tiles to [tile_size/2, tile_size/2], the tile_size seen by the network [default:True]
        TTA:            boolean, load Test Time Augmentation transformations [default:False]
        image_type:     str, one of 'mul', 'pan' [default: 'mul']
        mode:           str, one of 'eval', 'infer' [default: 'eval'], if set to infer
                        the dataloader will only contain images and tile_names (not labels)
        use_only_test_tiles: boolean, [default:False]
                        if set to False it will return all the data from the dataset.csv (i.e. 'train'/'val'/'test' splits)
                        if set to True it will only return the data from 'test' split 
    Usage:
        >>> experiment = "/home/minas/slumworld/data/output/experiments/SatelliteUnet/lightning_logs/version_18/"
        >>> checkpoint = experiment + "checkpoints/BinaryCrossEntropyWithLogits-AdamW--L2_2e-05-Seed_1357-TStamp_1629139845-epoch=225-val_loss=0.0606-val_acc=0.9757-val_f1=0.7871808409690857.ckpt'
        >>> dataset_file = "/home/minas/slumworld/data/tiled/MD_MUL_75_Briana/tiled_input/dataset.csv"
        >>> Loader = TestingDataLoader(dataset_file, batch_size=25, num_workers=4, tile_size=512, norm_file=norm_file)
        >>> testDataLoader = Loader.get_dataloader()
    '''
    def __init__(self, dataset_file=None, norm_file=None, batch_size=10, num_workers=1, tile_size=512, 
                 shuffle=False, foldID=-1, split_tiles=True, TTA=False, image_type='mul', mode='eval',
                 use_only_test_tiles=False):
        self.dataset_file = dataset_file
        self.foldID = foldID
        self.image_type = image_type
        self.tta = TTA
        self.mode = mode
        self.use_only_test_tiles = use_only_test_tiles
        df = pd.read_csv(dataset_file)
        if self.foldID == -1:
            training_column = "dataset_part"
        elif isinstance(self.foldID, int):
            training_column = f"Fold_{self.foldID}"
        if self.use_only_test_tiles:
            if len(df[df[training_column] == "Test"]) > 0:
                # dataset has been tiled with test-set, dataloader will use the Test data
                self.df = df[ (df[training_column] == "Test") | (df[training_column] == "Mask") ]
                print("Number of Test tiles:", sum(df[training_column] == "Test"))
            else:
                print("Parameter use_only_test_tiles is set to True, but no test data found in dataset.csv file.")
                print("Either re-run the script with use_only_test_tiles set to False, to test the model with all available tiles, or re-tile the image.")
                print("Aborting...")
                sys.exit(1)
        else:
            # will use all data
            self.df = df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tile_size = tile_size
        self.split_tiles = split_tiles
        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = data["mean"]
            self.std = data["std"]
        else:
            try:
                print("Normalization file not provided.Will use the dataset.json file in the data folder. Predictions prone to error.")
                with open(os.path.join(str(Path(dataset_file).parent), "dataset.json"), 'r') as json_file:
                    data = json.load(json_file)
                self.mean = data["mean"]
                self.std = data["std"]
            except Exception as Error:
                print("Could not find any normalization file.Will not normalize. Predictions prone to error.")
                self.mean = np.array([0.])
                self.std = np.array([1.])
        self.setup()

    def setup(self):
        if self.tta:
            transforms = INFERENCE_TRANSFORMS_DICT['tta'+'_'+self.image_type]
        else:
            transforms = INFERENCE_TRANSFORMS
        infer_transforms = create_transform(transforms, mean=self.mean, std=self.std, input_size=self.tile_size//2)
        self.test_dataset = ValidationDataset(validation_df=self.df, 
                                              transform=infer_transforms,
                                              tile_size=self.tile_size,
                                              split_tiles=self.split_tiles,
                                              mode=self.mode)

    def get_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size = self.batch_size, 
                                     num_workers=self.num_workers, shuffle=self.shuffle,
                                     pin_memory=False)
        return test_dataloader
