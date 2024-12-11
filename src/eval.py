import torch
from dataset import load_dataloader
from model import ResNetBigger, convnext_tiny, convnext_small, MelClassifier
from utils import load_config, ComputePerformance, visualize_result
from tqdm import tqdm


config = load_config('configs/eval_convnext.yml')
valid_dataloader = load_dataloader(**config['test'])
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

checkpoint = torch.load(config['checkpoint_path'])
model.load_state_dict(checkpoint['model_state_dict'])

total_score = dict()
all_predicted = []
all_labels = []

model.eval()

for batch in tqdm(valid_dataloader):
    inputs = batch['audio'].to(device)
    labels = batch['label'].to(device)

    inputs = inputs.unsqueeze(1)

    if config['model_type'] == 'convnext':
        inputs = torch.nn.functional.interpolate(inputs, size=(128, 160), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    _, predicted = outputs.max(1)
    all_predicted.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

score = ComputePerformance(all_labels, all_predicted)
visualize_result(score, config['save_dir'])