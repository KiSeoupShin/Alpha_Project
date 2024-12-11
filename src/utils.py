import yaml
from torch.utils.tensorboard import SummaryWriter
import os
import torchaudio
import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.optim as optim

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_metrics(config, model, optimizer, dataloader, attn_loss, backbone_loss, scheduler, type, device):
    total_loss = 0
    total_attn_loss = 0
    total_backbone_loss = 0
    predictions = []
    true_labels = []
    
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'{type}ing'):
        inputs = batch['audio'].to(device)
        labels = batch['label'].to(device)

        inputs = inputs.unsqueeze(1)

        if config['model_type'] == 'convnext':
            inputs = torch.nn.functional.interpolate(inputs, size=(128, 160), mode='bilinear', align_corners=False)
        
        if type == 'train':
            optimizer.zero_grad()
        
        attn_outputs, backbone_outputs = model(inputs)
        curr_attn_loss = attn_loss(attn_outputs, labels, model.parameters())
        curr_backbone_loss = backbone_loss(backbone_outputs, labels, model.parameters())
        loss = curr_attn_loss + curr_backbone_loss

        if type == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
        
        total_loss += loss.item()
        total_attn_loss += curr_attn_loss.item()
        total_backbone_loss += curr_backbone_loss.item()
        _, predicted = attn_outputs.max(1)
        
        # 예측값과 실제값을 리스트에 저장
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    avg_attn_loss = total_attn_loss / len(dataloader)
    avg_backbone_loss = total_backbone_loss / len(dataloader)
    
    # 전체 정확도 계산
    overall_accuracy = 100. * accuracy_score(true_labels, predictions)
    
    # 각 클래스별 정확도 계산
    class_report = classification_report(true_labels, predictions, output_dict=True)
    class_accuracies = {}
    
    for class_idx in range(config['backbone']['num_classes']):
        if str(class_idx) in class_report:
            class_accuracies[f'class_{class_idx}'] = 100. * class_report[str(class_idx)]['precision']
    
    # 균형 정확도 계산 (각 클래스의 정확도의 평균)
    balanced_acc = 100. * balanced_accuracy_score(true_labels, predictions)
    
    metrics = {
        'loss': avg_loss,
        'attn_loss': avg_attn_loss,
        'backbone_loss': avg_backbone_loss,
        'overall_accuracy': overall_accuracy,
        'balanced_accuracy': balanced_acc,
        'class_accuracies': class_accuracies
    }
    
    return metrics

def get_optim_and_criterion(model, initial_lr=1e-3, num_epochs=100, train_loader=None):
    from loss import FocalLoss
    
    attn_loss = FocalLoss()
    backbone_loss = FocalLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False
    )
    # scheduler = ReduceLROnPlateau(optimizer,
    #                               mode='min',
    #                               factor=0.5,
    #                               patience=20,
    #                               min_lr=1e-6,
    #                               threshold=1e-4,
    #                               cooldown=5
    #                               )    # 학습률 수렴 속도가 너무 빠름
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e-4
    )
    return attn_loss, backbone_loss, optimizer, scheduler

def create_writer(config):
    log_dir = config['log_dir']
    is_continue = config['is_continue']

    i = 0
    while os.path.exists(f"{log_dir}/logs_{i}"):
        i += 1
    if is_continue:
        new_log_dir = f"{log_dir}/logs_{i-1}/"
    else:
        new_log_dir = f"{log_dir}/logs_{i}/"
        os.makedirs(new_log_dir)
        os.makedirs(f"{new_log_dir}/ckpt/")
    
    # config를 yml 파일로 new_log_dir 폴더 안에 저장
    with open(os.path.join(new_log_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    return SummaryWriter(f"{new_log_dir}/tensorboard"), new_log_dir

def read_wav(data, base_sr):
        wav_path = data['wav']
        if 'start_time' in data and 'end_time' in data:
            start_time = data['start_time']
            end_time = data['end_time']
        else:
            start_time = None
            end_time = None    
        if start_time is not None and end_time is not None:
            sample_rate = torchaudio.info(wav_path).sample_rate
            frame_offset = int(start_time * sample_rate)
            num_frames = int(end_time * sample_rate) - frame_offset
            wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
        else:    
            wav, sr = torchaudio.load(wav_path)
        if sr != base_sr:
            wav = torchaudio.functional.resample(wav, sr, base_sr)
        wav = wav.reshape(-1)    
        return wav

def sampling_audio(audio, sr):
        if audio.shape[0] < 5 * sr:
            padding = torch.zeros(5 * sr - audio.shape[0])
            return torch.cat((audio, padding))

        start = torch.randint(0, audio.shape[0] - 5 * sr + 1, (1,)).item()
        audio = audio[start:start + 5 * sr]
        return audio
    

def featurize_melspec(y, sr, hop_length=None):
    S = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=hop_length)(y)
    S = torchaudio.functional.amplitude_to_DB(S, multiplier=10.0, amin=1e-10, db_multiplier=1.0)
    return S

def wav_to_mel(dataset_file, data_dir, save_dir, sr, hop_length):
    data_list = []

    metadata = json.load(open(os.path.join(data_dir, dataset_file, f'{dataset_file}.json')))
    data_list.extend([item for item in metadata.values()])

    save_dir = os.path.join(save_dir, dataset_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data in tqdm(data_list):
        mel_path = os.path.join(save_dir, *data['wav'].split('/')[2:-1], data['wav'].split('/')[-1].split('.')[0] + '.npy')
        if not os.path.exists(os.path.dirname(mel_path)):
            os.makedirs(os.path.dirname(mel_path))
        
        if not os.path.exists(mel_path):
            wav = read_wav(data, sr)
            wav = sampling_audio(wav, sr)
            mel = featurize_melspec(wav, sr, hop_length)
            np.save(mel_path, mel.cpu().numpy())

def ComputePerformance(ref_id, hyp_id):
    score = dict()
 
    num = len(ref_id)
    score['num'] = num
    score['overallWA'] = accuracy_score(ref_id, hyp_id)
    score['overallUA'] = balanced_accuracy_score(ref_id, hyp_id)
    score['overallMicroF1'] = f1_score(ref_id, hyp_id, average = 'micro')
    score['overallMacroF1'] = f1_score(ref_id, hyp_id, average = 'macro')
    score['report'] = classification_report(ref_id, hyp_id)
    score['confusion'] = confusion_matrix(ref_id, hyp_id)
    return score

def visualize_result(result, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_path = os.path.join(save_dir, "result.log")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Overall Weighted Accuracy: {result['overallWA']:.4f}\n")
        log_file.write(f"Overall Unweighted Accuracy: {result['overallUA']:.4f}\n")
        log_file.write(f"Overall Micro F1 Score: {result['overallMicroF1']:.4f}\n")
        log_file.write(f"Overall Macro F1 Score: {result['overallMacroF1']:.4f}\n")

        # # classification report를 데이터프레임으로 변환
        # report_data = []
        # lines = result['report'].split('\n')
        # for line in lines[2:10]:
        #     row = line.split()
        #     if len(row) > 0:
        #         report_data.append(row)
        # import ipdb; ipdb.set_trace()
        # report_df = pd.DataFrame(report_data, columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
        # report_df[["Precision", "Recall", "F1-Score", "Support"]] = report_df[["Precision", "Recall", "F1-Score", "Support"]].apply(pd.to_numeric)

        # # Classification Report 출력
        # log_file.write("\nClassification Report:\n")
        # log_file.write(report_df.to_string(index=False) + "\n")

        # confusion matrix 시각화
        confusion_df = pd.DataFrame(result['confusion'], index=[f"Class {i}" for i in range(len(result['confusion']))], columns=[f"Class {i}" for i in range(len(result['confusion']))])

        # Confusion Matrix 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))  # 이미지로 저장
        plt.close()  # 현재 플롯을 닫음