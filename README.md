# Alpha Project
**음성 감정 분류 모델 개발**

<br/>

## 1. 배경 & 목적
 
- wav 파일을 활용한 음성 감정 분류 모델은 매우 뛰어난 성능을 보여주고 있음
- 이를 wav 파일이 아닌 mel-spectrogram 에서도 뛰어난 모습을 보이는 모델을 개발하고자 함
- 이를 통해, 더욱 다양한 곳에서 활용이 가능한 모델을 만들고자 함

<br/>

## 2. 프로젝트 기간

- 2024년 09월 - 2024년 12월

<br/>

## 3. Model FLow

- Model Architecture
- ConvNext 모델과 mel-spectrogram을 통해 감정을 분류
- Positional Encoding 을 추가하여 시간적 정보를 파악할 수 있는지도 실험


## 4. Model Experiments
  
| Method                     | WA      | UA      | F1      |
|----------------------------|---------|---------|---------|
| ResNet                     | 0.5206  | 0.4940  | 0.4892  |
| ConvNext(Ours)             | **0.7310**  | **0.7279**  | **0.7279**  |
| ConvNext + PE(Ours)        | 0.5933  | 0.5794  | 0.5838  |

PE : Positional Encoding
WA : Weighted Accuracy, UA : Unweighted Accuracy, F1 : Macro F1 Score

<br/>

##

-참고 자료
[알파프로젝트_발표자료.pdf](https://github.com/user-attachments/files/18221908/_.pdf)

