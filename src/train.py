import torch
from dataset import load_dataloader
from model import ResNetBigger, convnext_tiny, convnext_small, MelClassifier
from utils import load_config, compute_metrics, create_writer, wav_to_mel, get_optim_and_criterion

# 설정 파일 로드
config = load_config('configs/train_convnext.yml')

# Mel Spectrogram 파일 생성
wav_to_mel(**config['mel'])

# 데이터로더 생성
train_dataloader = load_dataloader(**config['train'])
valid_dataloader = load_dataloader(**config['valid'])

# GPU 사용 가능 여부 확인
device = torch.device(config['device'])

# 모델 생성 및 GPU로 이동
# if config['model_type'] == 'resnet':
#     model = ResNetBigger(**config['model'])
# elif config['model_type'] == 'convnext':
#     if config['model_size'] == 'tiny':
#         model = convnext_tiny(**config['model'])
#     elif config['model_size'] == 'small':
#         model = convnext_small(**config['model'])
model = MelClassifier(config['backbone'], config['attention'])
model.to(device)

# checkpoint가 존재하면 불러오기
if config['is_continue']:
    # checkpoint 경로 설정
    checkpoint_path = config.get('checkpoint_path', None)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'체크포인트를 불러왔습니다. epoch {start_epoch}부터 학습을 시작합니다.')
else:
    start_epoch = 0

# Loss function과 optimizer 정의
attn_loss, backbone_loss, optimizer, scheduler = get_optim_and_criterion(model, 
                                                                         initial_lr=config['initial_lr'], 
                                                                         num_epochs=config['num_epochs'], 
                                                                         train_loader=train_dataloader
                                                                         )

# 체크포인트에서 optimizer와 scheduler 상태 불러오기
if config['is_continue']:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Tensorboard writer 생성
writer, log_dir = create_writer(config)

num_epochs = config['num_epochs']
best_loss = float('inf')

for epoch in range(start_epoch, start_epoch + num_epochs):
    # Training
    model.train()
    train_metrics = compute_metrics(config, 
                                    model, 
                                    optimizer, 
                                    train_dataloader, 
                                    attn_loss,
                                    backbone_loss,
                                    scheduler, 
                                    'train', 
                                    device
                                    )
    
    # Tensorboard에 기록
    writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
    writer.add_scalar('Loss/train/attn', train_metrics['attn_loss'], epoch)
    writer.add_scalar('Loss/train/backbone', train_metrics['backbone_loss'], epoch)
    writer.add_scalar('Accuracy/train', train_metrics['overall_accuracy'], epoch)
    writer.add_scalar('Accuracy/Balanced/train', train_metrics['balanced_accuracy'], epoch)
    writer.add_scalar('Learning Rate/train', optimizer.param_groups[0]['lr'], epoch)
    for class_idx, accuracy in train_metrics['class_accuracies'].items():
        writer.add_scalar(f'Accuracy/Class/train/{class_idx}', accuracy, epoch)
    
    print(f'Epoch {epoch} Training: Average Loss = {train_metrics["loss"]:.4f}, Accuracy = {train_metrics["overall_accuracy"]:.2f}%')

    # Validation
    model.eval()
    with torch.no_grad():
        val_metrics = compute_metrics(config, 
                                     model, 
                                     optimizer, 
                                     valid_dataloader, 
                                     attn_loss, 
                                     backbone_loss, 
                                     scheduler, 
                                     'validation', 
                                     device
                                     )

    # Tensorboard에 validation 결과 기록
    writer.add_scalar('Loss/valid', val_metrics['loss'], epoch)
    writer.add_scalar('Loss/valid/attn', val_metrics['attn_loss'], epoch)
    writer.add_scalar('Loss/valid/backbone', val_metrics['backbone_loss'], epoch)
    writer.add_scalar('Accuracy/valid', val_metrics['overall_accuracy'], epoch)
    writer.add_scalar('Accuracy/Balanced/valid', val_metrics['balanced_accuracy'], epoch)
    for class_idx, accuracy in val_metrics['class_accuracies'].items():
        writer.add_scalar(f'Accuracy/Class/valid/{class_idx}', accuracy, epoch)

    print(f'Epoch {epoch} Validation: Average Loss = {val_metrics["loss"]:.4f}, Accuracy = {val_metrics["overall_accuracy"]:.2f}%')
    
    # best loss의 모델 저장 (모든 학습 정보 포함)
    if val_metrics['loss'] < best_loss:
        best_loss = val_metrics['loss']
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{log_dir}/ckpt/best_model.pth')

writer.close()
