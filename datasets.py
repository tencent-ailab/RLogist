# standard library
import os
# 3rd part packages
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
# local source


class LoadStructData(object):
    def __init__(self, root_dir, sub_sets=None, pos_feature_coord_name='', neg_erase_feature_coord_name='',
                 neg_background_feature_coord_name=''):
        self.root_dir = root_dir
        self.sub_sets = sub_sets
        self.pos_feature_coord_name = pos_feature_coord_name
        self.neg_erase_feature_coord_name = neg_erase_feature_coord_name
        self.neg_background_feature_coord_name = neg_background_feature_coord_name
        self.feature_coord_dir = 'patch_coord_feature'
        self.feature_dir = 'patch_feature'
        self.label_dict = dict()
        self.slide_data = dict()
        self.load_class_label()
        self.load_dataset()

    def load_class_label(self):
        sub_set_dir = os.path.join(self.root_dir, self.sub_sets[0])

        label_id = 0
        for label_name in sorted(os.listdir(sub_set_dir)):
            self.label_dict[label_name] = label_id
            print(f'label_name: {label_name}, label_id: {label_id}')
            label_id += 1

    def load_dataset(self):
        for sub_set in self.sub_sets:
            sub_set_dir = os.path.join(self.root_dir, sub_set)

            if sub_set not in self.slide_data:
                self.slide_data[sub_set] = dict()
            self.slide_data[sub_set]['patient_id'] = list()
            self.slide_data[sub_set]['slide_id'] = list()
            self.slide_data[sub_set]['slide_label'] = list()
            self.slide_data[sub_set]['feature_name'] = list()
            self.slide_data[sub_set]['feature_coord_name'] = list()
            self.slide_data[sub_set]['sample_num'] = dict()
            self.slide_data[sub_set]['neg_feature_coord_name'] = list()
            self.slide_data[sub_set]['neg_feature_id'] = list()
            self.slide_data[sub_set]['pos_feature_coord_name'] = self.pos_feature_coord_name
            self.slide_data[sub_set]['neg_erase_feature_coord_name'] = self.neg_erase_feature_coord_name
            self.slide_data[sub_set]['neg_background_feature_coord_name'] = self.neg_background_feature_coord_name

            for label_name in os.listdir(sub_set_dir):
                whole_feature_dir = os.path.join(self.root_dir, sub_set, str(label_name), self.feature_dir)
                whole_feature_coord_dir = os.path.join(self.root_dir, sub_set, str(label_name), self.feature_coord_dir)
                label_sample_num = 0
                for slide_name in os.listdir(whole_feature_coord_dir):
                    slide_id = '.'.join(str(slide_name).split('.')[:-1])
                    if not str(slide_name).endswith('.h5'):
                        continue
                    patient_id = '_'.join(slide_id.split('_')[:2])

                    feature_pt_name = os.path.join(whole_feature_dir, slide_id + '.pt')
                    feature_coord_h5_name = os.path.join(whole_feature_coord_dir, slide_id + '.h5')

                    self.slide_data[sub_set]['patient_id'].append(patient_id)
                    self.slide_data[sub_set]['slide_id'].append(slide_id)
                    self.slide_data[sub_set]['slide_label'].append(self.label_dict[label_name])
                    self.slide_data[sub_set]['feature_name'].append(feature_pt_name)
                    self.slide_data[sub_set]['feature_coord_name'].append(feature_coord_h5_name)

                    if self.label_dict[label_name] == 0:
                        self.slide_data[sub_set]['neg_feature_coord_name'].append(feature_coord_h5_name)

                    label_sample_num += 1
                self.slide_data[sub_set]['sample_num'][self.label_dict[label_name]] = label_sample_num
        print(self.slide_data['train']['neg_feature_coord_name'])


class ClamDataset(Dataset):
    def __init__(self, slide_data, label_dict, use_h5=False):
        self.slide_data = slide_data
        self.num_class = len(label_dict)
        self.use_h5 = use_h5
        self.pos_feature_coord_name = self.slide_data['pos_feature_coord_name']
        self.neg_erase_feature_coord_name = self.slide_data['neg_erase_feature_coord_name']
        self.neg_background_feature_coord_name = self.slide_data['neg_background_feature_coord_name']
        self.pos_feature = self.load_pos_feature()
        self.neg_erase_feature = self.load_neg_erase_feature()
        self.neg_background_feature = self.load_neg_background_feature()

    def load_pos_feature(self):

        with h5py.File(self.pos_feature_coord_name, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        features = torch.from_numpy(features)
        return features

    def load_neg_erase_feature(self):
        with h5py.File(self.neg_erase_feature_coord_name, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        features = torch.from_numpy(features)
        return features

    def load_neg_background_feature(self):
        with h5py.File(self.neg_background_feature_coord_name, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        features = torch.from_numpy(features)
        return features

    def load_neg_feature(self, h5_name):
        with h5py.File(h5_name, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]
            features = features[:, 0, :]

        neg_erase_select_num = 300
        neg_erase_select_index = np.random.choice(
            range(self.neg_erase_feature.shape[0]), size=(neg_erase_select_num, 1), replace=False)
        select_neg_erase_feature = np.take_along_axis(self.neg_erase_feature, neg_erase_select_index, axis=0)

        neg_background_select_num = 300
        neg_background_select_index = np.random.choice(
            range(self.neg_background_feature.shape[0]), size=(neg_background_select_num, 1), replace=False)
        select_neg_background_feature = np.take_along_axis(
            self.neg_background_feature, neg_background_select_index, axis=0)

        cat_features = np.concatenate((features, select_neg_erase_feature, select_neg_background_feature), axis=0)

        cat_features = torch.from_numpy(cat_features)
        return cat_features

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['slide_label'][idx]
        # print(f'slide_id: {slide_id}, label: {label}')

        if not self.use_h5:

            feature_pt_name = self.slide_data['feature_name'][idx]
            features = torch.load(feature_pt_name)
            return features, label

        else:
            feature_h5_name = self.slide_data['feature_coord_name'][idx]

            with h5py.File(feature_h5_name, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
                features = features[:, 0, :]

            features = torch.from_numpy(features)

            return features, label, coords

    def __len__(self):
        return len(self.slide_data['slide_id'])


def get_split_dataset(
        root_dir, sub_sets, use_h5=False, pos_feature_coord_name='',
        neg_erase_feature_coord_name='', neg_background_feature_coord_name=''):
    load_data_obj = LoadStructData(
        root_dir=root_dir,
        sub_sets=sub_sets,
        pos_feature_coord_name=pos_feature_coord_name,
        neg_erase_feature_coord_name=neg_erase_feature_coord_name,
        neg_background_feature_coord_name=neg_background_feature_coord_name)

    train_dataset = ClamDataset(
        slide_data=load_data_obj.slide_data['train'],
        label_dict=load_data_obj.label_dict,
        use_h5=use_h5)
    val_dataset = ClamDataset(
        slide_data=load_data_obj.slide_data['val'],
        label_dict=load_data_obj.label_dict,
        use_h5=use_h5)
    test_dataset = ClamDataset(
        slide_data=load_data_obj.slide_data['test'],
        label_dict=load_data_obj.label_dict,
        use_h5=use_h5)
    return train_dataset, val_dataset, test_dataset
