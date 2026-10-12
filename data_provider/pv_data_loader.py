"""
PV Data Loader for MultdBG-Time-Series-Library

This module provides data loaders for PV forecasting datasets that have been
converted from the transfer-learning-PV-forecasting format to MultdBG format.

Key features:
- Handles site-based splits (different sites for train/val/test)
- Supports both time-based and site-based split configurations
- Maintains site_id tracking for evaluation
- Compatible with MultdBG's existing data loader interface
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class Dataset_PV_Custom(Dataset):
    """
    Custom dataset class for PV forecasting data in MultdBG format
    
    This class handles PV datasets that have been converted from the 
    transfer-learning-PV-forecasting format, supporting both site-based
    and time-based splits.
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='DKASC_AliceSprings.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        Initialize PV dataset
        
        Args:
            args: Arguments object containing configuration
            root_path: Root path to dataset directory
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M', 'MS', or 'S' (same as MultdBG)
            data_path: CSV filename
            target: Target column name (default: 'OT')
            scale: Whether to apply scaling
            timeenc: Time encoding method (0 or 1)
            freq: Time frequency
            seasonal_patterns: Seasonal patterns (not used for PV data)
        """
        
        if size is None:
            self.seq_len   = 96 # 24 * 4 * 4
            self.label_len = 48 # 24 * 4
            self.pred_len  = 48 # 24 * 4
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]
        
        assert flag in ['train', 'test', 'val']
        
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        # load data and split configuration
        self.data_path = os.path.join(root_path, data_path)
        self.dataset_name = data_path.replace('.csv', '')
        
        # load split configuration
        self.split_config = self._load_split_config()
        
        # load site information
        self.site_info = self._load_site_info()
        
        # load and process data
        self._read_data()
    
    def _load_split_config(self):
        # load split configuration from JSON file
        split_file = os.path.join(self.root_path, f"{self.dataset_name}_splits.json")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return json.load(f)
        else:
            # default time-based split if no config found
            return {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    def _load_site_info(self):
        # load site information from JSON file
        site_file = os.path.join(self.root_path, f"{self.dataset_name}_site_info.json")
        if os.path.exists(site_file):
            with open(site_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _read_data(self):
        # load and process the data according to split configuration
        
        # load the consolidated CSV file
        df_raw = pd.read_csv(self.data_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # determine if this is site-based or time-based split
        is_site_based = isinstance(list(self.split_config.values())[0], list)
        
        if is_site_based:
            # site-based split: filter by site_id
            target_sites = self.split_config[self.flag]
            df_raw = df_raw[df_raw['site_id'].isin(target_sites)]
            print(f"Site-based split - {self.flag}: {len(target_sites)} sites, {len(df_raw)} data points")
        else:
            # time-based split: split by time
            total_len = len(df_raw)
            train_end = int(total_len * self.split_config['train'])
            val_end = train_end + int(total_len * self.split_config['val'])
            
            if self.flag == 'train':
                df_raw = df_raw[:train_end]
            elif self.flag == 'val':
                df_raw = df_raw[train_end:val_end]
            else:  # test
                df_raw = df_raw[val_end:]
            
            print(f"Time-based split - {self.flag}: {len(df_raw)} data points")
        
        # prepare features based on features parameter
        if self.features == 'M' or self.features == 'MS':
            # use all features except date and site_id
            feature_cols = [col for col in df_raw.columns if col not in ['date', 'site_id']]
            df_data = df_raw[feature_cols]
        elif self.features == 'S':
            # use only target feature
            df_data = df_raw[[self.target]]
        
        # apply scaling if requested
        if self.scale:
            self.scaler = StandardScaler()
            scaler_path = os.path.join(self.root_path, f"{self.dataset_name}_{self.features}_scaler.pkl")
            if self.flag == 'train':
                # fit scaler on training data
                self.scaler.fit(df_data.values)
                # save scaler for later use
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                # load existing scaler
                import pickle
                loaded_scaler = None
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        loaded_scaler = pickle.load(f)
                if loaded_scaler is None:
                    print(f"Warning: Scaler not found at {scaler_path}, proceeding without scaling")
                    self.scaler = None
                else:
                    # verify feature count matches; if not, refit on training split to avoid leakage
                    expected_features = df_data.shape[1]
                    n_features_in = getattr(loaded_scaler, 'n_features_in_', None)
                    if n_features_in is not None and n_features_in != expected_features:
                        print(
                            f"Warning: Loaded scaler expects {n_features_in} features but data has {expected_features}. "
                            "Refitting scaler on training split to match current features."
                        )
                        # Build training split and fit a new scaler with the same feature selection
                        df_full = pd.read_csv(self.data_path)
                        df_full['date'] = pd.to_datetime(df_full['date'])
                        is_site_based_train = isinstance(list(self.split_config.values())[0], list)
                        if is_site_based_train:
                            train_sites = self.split_config.get('train', [])
                            df_train = df_full[df_full['site_id'].isin(train_sites)]
                        else:
                            total_len_full = len(df_full)
                            train_end_full = int(total_len_full * self.split_config['train'])
                            df_train = df_full[:train_end_full]
                        if self.features == 'M' or self.features == 'MS':
                            train_feature_cols = [col for col in df_train.columns if col not in ['date', 'site_id']]
                            df_train_data = df_train[train_feature_cols]
                        else:
                            df_train_data = df_train[[self.target]]
                        self.scaler = StandardScaler()
                        self.scaler.fit(df_train_data.values)
                        # overwrite persisted scaler to avoid future mismatch
                        try:
                            with open(scaler_path, 'wb') as f:
                                pickle.dump(self.scaler, f)
                        except Exception as e:
                            print(f"Warning: Failed to overwrite scaler at {scaler_path}: {e}")
                    else:
                        self.scaler = loaded_scaler
            
            if self.scaler is not None:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
        else:
            data = df_data.values
        
        # prepare time features
        df_stamp = df_raw[['date']].copy()
        if self.timeenc == 0:
            # simple time features
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day']   = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # advanced time features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # store data
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
        # store site IDs for evaluation
        self.site_ids = df_raw['site_id'].values if 'site_id' in df_raw.columns else None
        
        print(f"Data loaded - X shape: {self.data_x.shape}, Y shape: {self.data_y.shape}")
        
        # Validate that no cross-site contamination will occur
        if self.site_ids is not None:
            self._validate_site_continuity()
    
    def __getitem__(self, index):
        # get a single sample from the dataset
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # ensure sequences don't cross site boundaries
        if self.site_ids is not None:
            current_site = self.site_ids[s_begin]
            
            # find the start and end of this site's data block
            site_indices = np.where(self.site_ids == current_site)[0]
            site_start = site_indices[0]
            site_end = site_indices[-1] + 1
            
            # ensure input sequence doesn't cross site boundaries
            if s_end > site_end:
                # Adjust sequence to fit within site boundaries
                s_begin = max(site_start, site_end - self.seq_len)
                s_end = site_end
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
            
            # ensure target sequence doesn't cross site boundaries
            if r_end > site_end:
                # Adjust target sequence to fit within site boundaries
                r_end = site_end
                r_begin = max(site_start, r_end - self.label_len - self.pred_len)
                s_end = r_begin + self.label_len
                s_begin = max(site_start, s_end - self.seq_len)
            
            # final check to ensure enough data for the sequence
            if s_end - s_begin < self.seq_len or r_end - r_begin < self.label_len + self.pred_len:
                # if  can't create a valid sequence, return a different valid sample
                # find the next valid sequence within the same site
                next_valid_index = self._find_next_valid_sequence(site_start, site_end)
                if next_valid_index is not None:
                    return self.__getitem__(next_valid_index)
                else:
                    # return the last valid sequence for this site
                    s_begin = max(site_start, site_end - self.seq_len)
                    s_end = site_end
                    r_begin = max(site_start, s_end - self.label_len)
                    r_end = min(site_end, r_begin + self.label_len + self.pred_len)
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # store site_id for this sample (for evaluation if needed)
        if self.site_ids is not None:
            site_id = self.site_ids[s_begin] if s_begin < len(self.site_ids) else 0
            # store the site_id as an attribute for this specific sample
            # this can be accessed later if needed for evaluation
            self._current_site_id = site_id
        
        # always return 4 values to match the expected format
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def _find_next_valid_sequence(self, site_start, site_end):
        # find the next valid sequence index within the same site
        
        #site_start: start index of the current site's data
        #site_end: end index of the current site's data

        # look for a valid sequence within the site boundaries
        for idx in range(site_start, site_end - self.seq_len - self.pred_len + 1):
            # check if this index would create a valid sequence
            s_begin = idx
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            # check if all indices are within site boundaries
            if (s_begin >= site_start and s_end <= site_end and 
                r_begin >= site_start and r_end <= site_end):
                return idx
        
        return None
    
    def _validate_site_continuity(self):
        # validate that sequences won't have cross-site contamination

        print(f"\n=== Validating Site Continuity for {self.flag} split ===")
        
        # check for site transitions in the data
        site_transitions = []
        prev_site = None
        for idx, site_id in enumerate(self.site_ids):
            if prev_site is not None and site_id != prev_site:
                site_transitions.append((idx, prev_site, site_id))
            prev_site = site_id
        
        print(f"Found {len(site_transitions)} site transitions")
        
        # check each site's data block
        unique_sites = np.unique(self.site_ids)
        for site_id in unique_sites:
            site_indices = np.where(self.site_ids == site_id)[0]
            site_start = site_indices[0]
            site_end = site_indices[-1] + 1
            site_length = site_end - site_start
            
            print(f"Site {site_id}: {site_length} data points (indices {site_start}-{site_end-1})")
            
            # check if this site has enough data for sequences
            min_required = self.seq_len + self.pred_len
            if site_length < min_required:
                print(f"  == WARNING: Site {site_id} has only {site_length} points, but needs {min_required} for sequences")
            else:
                max_sequences = site_length - self.seq_len - self.pred_len + 1
                print(f"  Can generate {max_sequences} valid sequences")
        
        print("== Site continuity validation completed")
    
    def __len__(self):
        # return the number of samples in the dataset
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        # inverse transform scaled data back to original scale
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def get_site_ids(self):
        # return site IDs for all samples in the dataset
        # this can be used for evaluation purposes
        if self.site_ids is not None:
            # return site IDs for all possible samples
            return self.site_ids[:-self.seq_len - self.pred_len + 1]
        else:
            return None

# dataset mapping for easy integration with existing MultdBG code
pv_data_dict = {
    'DKASC_AliceSprings': Dataset_PV_Custom,
    'DKASC_Yulara': Dataset_PV_Custom,
    'GIST': Dataset_PV_Custom,
    'Germany': Dataset_PV_Custom,
    'Miryang': Dataset_PV_Custom,
    'OEDI_California': Dataset_PV_Custom,
    'OEDI_Georgia': Dataset_PV_Custom,
    'UK': Dataset_PV_Custom,
}
