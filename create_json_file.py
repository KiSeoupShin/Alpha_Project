import os
from EmoBox.EmoDataset import EmoDataset_Downloader
from tqdm import tqdm

def main():
    # 설정
    base_dir = "/home/shinkiseoup/Desktop/AlphaProject/"
    data_dir = os.path.join(base_dir, "data/")  # 데이터가 저장된 경로

    dataset_list = os.listdir(data_dir)

    for dataset in tqdm(dataset_list):
        if dataset in ["casia", "m3ed", "mer2023", "msppodcast", "track2"]:
            continue    

        print(f"Start processing {dataset}")

        if dataset == "emovo":
            meta_data_dir = os.path.join(base_dir, f"data/{dataset}/{dataset}.jsonl")  # 메타데이터 경로
            label_map_path = os.path.join(base_dir, f"data/{dataset}/label_dict.json")  # 라벨 맵 경로
        else:
            meta_data_dir = os.path.join(base_dir, f"data/{dataset}/{dataset}.jsonl")  # 메타데이터 경로
            label_map_path = os.path.join(base_dir, f"data/{dataset}/label_map.json")  # 라벨 맵 경로

        fold = '1'

        # 데이터셋 생성
        train_dataset = EmoDataset_Downloader(dataset=dataset, data_dir=base_dir, meta_data_dir=meta_data_dir, label_map=label_map_path, fold=fold, split="train")
        valid_dataset = EmoDataset_Downloader(dataset=dataset, data_dir=base_dir, meta_data_dir=meta_data_dir, label_map=label_map_path, fold=fold, split="valid")
        test_dataset = EmoDataset_Downloader(dataset=dataset, data_dir=base_dir, meta_data_dir=meta_data_dir, label_map=label_map_path, fold=fold, split="test")

        save_base_dir = os.path.join(base_dir, 'json_file', dataset)

        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir, exist_ok=True)

        train_dataset.save_dataset_to_json(os.path.join(save_base_dir, f'train_fold_{fold}.json'))
        valid_dataset.save_dataset_to_json(os.path.join(save_base_dir, f'valid_fold_{fold}.json'))
        test_dataset.save_dataset_to_json(os.path.join(save_base_dir, f'test_fold_{fold}.json'))

        # 데이터셋 정보 출력
        print(f"Train dataset size for fold {fold}: {len(train_dataset)}")
        print(f"Valid dataset size for fold {fold}: {len(valid_dataset)}")
        print(f"Test dataset size for fold {fold}: {len(test_dataset)}")

if __name__ == "__main__":
    main()