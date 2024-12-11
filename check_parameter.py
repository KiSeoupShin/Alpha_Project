import torch
from src.model import ResNetBigger, convnext_tiny, convnext_small
from src.utils import load_config

# 설정 파일 로드
config = load_config('configs/train_convnext.yml')

# 모델 생성
# model = ResNetBigger(**config['model'])
model = convnext_tiny(**config['model'])

# 전체 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'전체 파라미터 수: {total_params:,}')
print(f'학습 가능한 파라미터 수: {trainable_params:,}')

# 각 레이어별 파라미터 수 출력
print('\n레이어별 파라미터 수:')
for name, param in model.named_parameters():
    print(f'{name}: {param.numel():,}')
