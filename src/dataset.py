from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torch
import json
import os
import numpy as np

def load_dataloader(dataset_type,
                    process_type,
                    data_dir, 
                    json_dir, 
                    mel_dir,
                    batch_size, 
                    shuffle,
                    dataset_file=None):
    
    if dataset_type == "single":
        dataset = SingleEmoDataset(json_dir, 
                                   data_dir, 
                                   mel_dir,
                                   dataset_file, 
                                   process_type)
    elif dataset_type == "multi":
        dataset = EmoDataset(json_dir, 
                             data_dir, 
                             mel_dir, 
                             process_type)
    else:
        raise ValueError("Invalid dataset type: single or multi.")
    
    return DataLoader(dataset, 
                      num_workers=20, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      collate_fn=collate_fn)

def collate_fn(batch):
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        return None
        
    audio_batch = torch.stack([item['audio'] for item in valid_batch])
    label_batch = torch.tensor([item['label'] for item in valid_batch])
    
    return {
        'audio': audio_batch,
        'label': label_batch
    }


class EmoDataset(Dataset):
    def __init__(
            self, 
            json_dir, 
            data_dir, 
            mel_dir,
            process_type
        ):
        super().__init__()
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.mel_dir = mel_dir
        self.data_list = []
        self.label_to_index = {}

        json_file = "train_fold_1.json" if process_type == "train" else "valid_fold_1.json" if process_type == "valid" else "test_fold_1.json"

        for dataset_file in os.listdir(self.json_dir):
            data = json.load(open(os.path.join(self.json_dir, dataset_file, json_file)))['data']
            self.data_list.extend([item for item in data if 'pavoque' not in item['wav']])

            if dataset_file == "emovo":
                label_map_path = os.path.join(data_dir, 'data', f"{dataset_file}/label_dict.json")
            else:
                label_map_path = os.path.join(data_dir, 'data', f"{dataset_file}/label_map.json")

            # 라벨을 숫자로 인코딩하기 위한 딕셔너리 생성
            label_map = json.load(open(label_map_path))
            for label, idx in enumerate(label_map.values()):
                if label not in self.label_to_index:
                    self.label_to_index[label] = idx
    

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_path = os.path.join(self.mel_dir, f"{data['wav'].split('/')[-1]}".replace('.wav', '.npy'))
        if not os.path.exists(mel_path):
            return None

        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()

        # 라벨을 숫자로 인코딩
        label = self.label_to_index[data['emo']]
        
        return {
            "audio": mel,
            "label": label,
        }
    

class SingleEmoDataset(Dataset):
    def __init__(
            self, 
            json_dir,
            data_dir, 
            mel_dir,
            dataset_file,
            process_type
        ):
        super().__init__()
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.mel_dir = mel_dir
        self.dataset_file = dataset_file
        self.data_list = []
        self.label_to_index = {}

        json_file = "train_fold_1.json" if process_type == "train" else "valid_fold_1.json" if process_type == "valid" else "test_fold_1.json"

        data = json.load(open(os.path.join(self.json_dir, dataset_file, json_file)))['data']
        self.data_list.extend([item for item in data])

        if dataset_file == "emovo":
            label_map_path = os.path.join(data_dir, 'data', f"{dataset_file}/label_dict.json")
        else:
            label_map_path = os.path.join(data_dir, 'data', f"{dataset_file}/label_map.json")

        # 라벨을 숫자로 인코딩하기 위한 딕셔너리 생성
        label_map = json.load(open(label_map_path))
        for idx, label in enumerate(label_map.values()):
            if label not in self.label_to_index:
                self.label_to_index[label] = idx
    

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_path = os.path.join(self.mel_dir, self.dataset_file, *data['wav'].split('/')[2:-1], data['wav'].split('/')[-1].split('.')[0] + '.npy')
        if not os.path.exists(mel_path):
            return None

        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()

        # 라벨을 숫자로 인코딩
        label = self.label_to_index[data['emo']]
        
        return {
            "audio": mel,
            "label": label,
        }