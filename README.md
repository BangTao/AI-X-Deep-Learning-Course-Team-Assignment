# AI-X-Deep-Learning-Course-Team-Assignment
AI+X: Deep Learning Course Team Assignment

Title:건설 현장 안전 강화: 건설 현장에서 YOLOv5 기반의 딥 러닝 안전 헬멧 감지 

Members: 
（1）하문월, Department: 무용학부, Email: 318yue@naver.com ;
（2）사경안, Department:경제금융학부, Email: xlq945270@163.com
（3）이서, Department:건설환경공학, Email: lirui1314-@naver.com;
（4）사봉개, Department:기계공학부, Email: bb13639806638@163.com;

 I. Proposal (Option 1)  
 - Motivation: Why are you doing this?
 - 저희 팀 멤버 중에는 건설 업계에 종사하는 가족을 둔 중국인 학생 이서와 사경안이 있으며, 이서 학생은 또한 한양대학교 건설환경공학 전공입니다. 그래서 주제를 정할 때, 건설 작업자의 안전에 초점을 맞추고자 하였습니다. 관련 자료를 검토한 결과, 중국에서 건설업은 모든 산업 중 사망자 수가 가장 많은 산업이며, 사망한 근로자의 5분의 1은 건설업에서 발생한다는 것을 알게 되었습니다. 근로자의 가족, 산업, 국가 모두 큰 손실을 입고 있습니다. 중국에서 진행 중인 건설 프로젝트의 수가 많고 계속 증가하고 있기 때문에, 건설 현장 근로자의 안전을 자동으로 모니터링할 혁신적인 방법을 개발할 필요성이 더욱 커지고 있습니다. 머리는 인체에서 가장 중요한 부분으로 충격을 가장 쉽게 받아 심각한 부상이나 사망을 초래하기 때문에, 건설 작업 중에는 보호 헬멧을 사용해야 합니다. 본 연구에서는 건설 현장의 감시 카메라 영상을 분석하여 건설 작업자가 안전 헬멧을 착용하고 있는지 자동으로 감지하는 것을 목표로 합니다.

 - What do you want to see at the end?
 - 실시간 비디오 피드에서 건설 작업자의 안전 헬멧 착용 여부를 정확하게 탐지할 수 있는 신뢰할 수 있는 시스템 개발을 목표로 합니다. 프로젝트가 끝나면, 건설 현장 작업자가 안전 헬멧을 착용하고 있는지 식별하는 YOLOv5 딥 러닝 모델의 효과를 보여주는 기능하는 프로토타입을 갖게 되기를 기대합니다. 이 시스템이 현장 안전 프로토콜을 강화하는 잠재적 도구로서 활용되어 머리 부상 위험을 줄이고 건설 산업의 전반적인 안전 관리에 기여할 수 있기를 바랍니다.
   
II. Datasets – 이 부분부터는 나중에 하셔도 됩니다. 

**(1) 데이터셋 출처**: 저희 프로젝트에 사용된 "Hard Hat Detection" 데이터셋은 Kaggle에서 다음 링크를 통해 접근할 수 있습니다: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection/.

(2) 데이터셋 개요: 이 데이터셋은 작업장 안전을 개선하기 위해 5,000개의 경계 상자 주석이 포함된 이미지를 포함하며, 헬멧, 사람 및 머리 등 세 가지 클래스로 분류됩니다.

(3) 데이터 주석 변환:
 - 원본 데이터셋은 PASCAL VOC 형식의 경계 상자 주석을 이미 제공합니다.
 - 데이터 주석에 대해 더 깊이 학습하고자 우리는 기존의 주석 정보를 사용하지 않고 원본 이미지만을 사용하여 직접 재주석을 진행하기로 결정했습니다.
 - labelimg 도구를 사용하여 'hat'과 'person' 두 클래스에 대한 이미지 재주석을 완료했습니다.
 - 주석을 완료한 후, YOLOv5가 인식할 수 있는 txt 형식으로 VOC 형식의 주석 정보를 변환했습니다. 이 변환 과정은 **prepare_data.py** 스크립트를 사용하여 완료되었으며, 구체적인 코드 및 작업은 GitHub 저장소 https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment에서 확인할 수 있습니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/add3ed45-df39-48f6-85c3-130d6a08509e)

(4) 데이터셋 분할: YOLOv5 모델을 훈련하기 전에 데이터셋을 훈련 세트와 검증 세트로 분할해야 합니다. 이 단계 또한 prepare_data.py 스크립트를 통해 구현되었으며, 데이터의 정확한 분할과 모델 훈련의 효율성을 보장합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/1a1acbf1-3912-4048-a8a0-d8b3564cfb53)


III. Methodology -- YOLOv5 알고리즘

YOLOv5는 You Only Look Once (YOLO) 시리즈의 최신 실시간 객체 탐지 알고리즘입니다. 그 속도와 정확도로 유명하며, 자원이 제한된 환경에서의 실시간 응용 프로그램에 특히 적합합니다. YOLOv5는 객체 분류와 위치 지정을 위한 단일 합성곱 신경망(CNN) 패스를 사용합니다. 멀티 스케일 예측과 Path Aggregation Network (PANet)을 통해 탐지 정확도를 향상시켰습니다. 또한, 개선된 앵커 메커니즘, 손실 함수 및 데이터 증강 전략을 도입하여 성능을 더욱 향상시켰습니다. YOLOv5의 이러한 특성들은 건설 안전 모니터링과 같이 빠르고 정확한 탐지가 요구되는 응용 프로그램에 매우 적합합니다.

（1） 환경 설정 및 사전 훈련된 가중치 가져오기

1.1 프로젝트 클론

YOLOv5는 오픈 소스이므로 GitHub에서 직접 소스 코드를 클론할 수 있습니다. 먼저 YOLOv5의 GitHub 페이지에 방문하여 프로젝트를 다운로드하십시오. https://github.com/ultralytics/yolov5

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/c645abb3-0c9a-405a-8766-516469fc8468)

1.2 프로젝트 코드 구조

YOLOv5 ZIP 파일을 다운로드하고 압축을 푼 후, PyCharm IDE를 통해 프로젝트를 엽니다. 프로젝트의 주요 디렉토리 구조는 다음과 같습니다:

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/5ddd3a64-2fb8-44ae-adb4-5c43b09341ae)

 - **data**: 학습, 테스트, 검증 데이터셋의 경로를 설정하는 설정 파일(yaml 파일)이 포함되어 있습니다. 이 파일에는 탐지할 대상의 카테고리 수와 이름이 포함되어 있습니다. 또한, 공식 테스트 이미지도 포함되어 있습니다. 자체 데이터셋을 훈련하는 경우 해당 yaml 파일을 수정해야 합니다.
 - **models**: 다양한 네트워크 구성과 함수를 포함하고 있으며, s, m, l, x 네 가지 버전의 모델이 포함되어 있습니다. 이 이름들은 모델의 크기를 나타냅니다. 속도는 빠른 것부터 느린 것까지, 정확도는 낮은 것부터 높은 것까지 다릅니다. 자체 데이터셋을 훈련하는 경우 해당 yaml 파일을 수정하여 모델을 훈련해야 합니다.
 - **Utils**: 손실 함수, 측정 지표 함수, 그래프 함수 등 다양한 유틸리티 함수가 포함되어 있습니다.
 - **weights**: 훈련된 가중치 파일이 저장되어 있습니다.
 - **detect.py**: 훈련된 가중치를 사용하여 이미지, 비디오, 카메라에서 목표를 탐지하는 스크립트입니다.
 - **train.py**: 데이터셋을 훈련하는 함수입니다.
 - **test.py**: 훈련 결과를 테스트하는 함수입니다.
 - **requirements.txt**: 프로젝트에 필요한 환경 의존성 패키지 및 버전이 나열되어 있어, 이 파일을 통해 필요한 라이브러리를 빠르게 설치할 수 있습니다.

1.3 환경 및 의존성 설치

PyCharm 커맨드 라인 터미널에서 다음 명령어를 입력하여 YOLOv5에 필요한 환경 및 의존성 라이브러리를 설치합니다:
pip install -r requirements.txt

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/a1f0d19e-e609-4218-bb9d-082a4131faaa)

1.4 데이터 표시 및 분할 작업

데이터 표시 단계에서 처리한 데이터 세트를 모델 훈련에 계속 사용합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/95c6477c-867f-4d04-9d4a-e59fe48e1b4f)

1.5 사전 훈련된 가중치 획득

강의에서 배운 것처럼, 신경망 모델의 효과는 가중치 매개변수에 크게 의존합니다. 모델 훈련을 가속화하고 정확도를 높이기 위해, 우리는 훈련에 사전 훈련된 가중치를 로드합니다. YOLOv5는 여러 사전 훈련된 가중치 버전을 제공하며, 우리는 요구 사항에 따라 적절한 사전 훈련된 가중치를 선택합니다. 이 프로젝트에서 우리는 공식적으로 제공된 yolov5s.pt 가중치 파일을 시작점으로 선택했습니다. 사전 훈련된 가중치는 이 링크에서 다운로드할 수 있습니다. （ https://github.com/ultralytics/yolov5）

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/c62c8b92-c5f5-4945-a7f7-6c2d255d8d54)

**（2）자체 모델 훈련**

환경 구축, 의존성 설치, 데이터셋 준비, 사전 훈련된 가중치 확보 등의 준비 작업을 마친 후, 이제 YOLOv5 목표 검출 모델의 훈련을 시작할 수 있습니다. 다음은 구체적인 훈련 단계입니다:

2.1 데이터 구성 파일 수정 

먼저, 특정 요구 사항에 맞게 데이터 구성 파일을 조정해야 합니다. data 디렉토리에서 voc.yaml 파일을 찾아 우리 데이터셋 구조와 클래스에 맞게 수정합니다. 완료 후, 이 파일을 hat.yaml이라는 이름으로 저장합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/8e4ab9bc-fd96-4b47-98cc-e30e69b3ff09)

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/c575d3f9-d33f-482d-9ee0-2cc86566853b)

2.2 모델 구성 파일 수정

사전 훈련된 가중치로 yolov5s.pt를 선택했기 때문에, 모델 구성 파일을 수정해야 합니다. models 디렉토리에서 yolov5s.yaml 파일을 찾아 프로젝트의 구체적 요구 사항과 일치하도록 필드를 업데이트합니다. 수정 후, 이 파일을 yolov5s_hat.yaml로 저장합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/b522e2d5-0826-423d-b25a-be306959f119)

2.3 모델 훈련 시작

구성 파일을 설정한 후, 모델 훈련 단계로 넘어갈 수 있습니다. train.py 파일을 열고 메인 함수 입구를 찾아 데이터셋 경로, hat.yaml 파일 및 yolov5s_hat.yaml 파일의 위치에 따라 해당 훈련 매개변수를 조정합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/aeec1eea-26b1-4b93-9c6d-acc2d0e949f4)

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/e2749b9e-fc12-420f-9e6d-2a48e771045c)

2.4 훈련 라운드 조정 훈련 과정을 더 잘 제어하고 모델의 성능을 적시에 관찰할 수 있도록 기본 훈련 라운드를 300회에서 100회로 조정합니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/efb66ae7-1c82-4ecb-932f-eb1d6f88744f)

이 단계를 마치면 train.py 스크립트를 실행하여 훈련 과정을 시작할 수 있습니다. 훈련 진행 상황과 성능 지표는 실시간으로 표시되어 모델의 학습 상태를 모니터링할 수 있습니다.

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/0816a989-e260-4a05-8550-4a20a2954100)

IV. Evaluation & Analysis - 

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/37cd5436-3f88-4a4a-a29a-00603b72d281)

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/5565e472-eeec-4e49-964e-066f77c14ed0)

![image](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/7d6c5a0c-7428-4895-bd2e-7adccd8ce0fb)

![F1_curve](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/75e96ef0-7a5e-4eb6-8f8c-f89ffee43af9)

![confusion_matrix](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/09e89632-a3a9-4513-8fc9-134c2b32a228)

![results](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/23730313-8d99-450c-b7db-e81695265b20)

![train_batch0](https://github.com/BangTao/AI-X-Deep-Learning-Course-Team-Assignment/assets/42737652/f2d9d46c-3fe1-4111-823e-0bcfe5a6c10d)


V. Related Work (e.g., existing studies) - Tools, libraries, blogs, or any documentation that you have used to do this project. 

- (1)관련 논문：
A.	H. M. Rubaiyat et al., "Automatic Detection of Helmet Uses for Construction Safety," 2016 IEEE/WIC/ACM International Conference on Web Intelligence Workshops (WIW), Omaha, NE, USA, 2016, pp. 135-142, doi: 10.1109/WIW.2016.045.
B.	Li, Y., Wei, H., Han, Z., Huang, J., & Wang, W. (2020). Deep Learning-Based Safety Helmet Detection in Engineering Management Based on Convolutional Neural Networks. Advances in Civil Engineering, 2020, 9703560. https://doi.org/10.1155/2020/9703560

-(2)Libraries：
	https://github.com/ultralytics/yolov5；
	https://developer.nvidia.com/cuda-toolkit
 
-(3)IDE： https://www.jetbrains.com/zh-cn/pycharm/

VI. Conclusion: Discussion



