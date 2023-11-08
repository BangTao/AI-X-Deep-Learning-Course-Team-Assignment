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

III. Methodology -- YOLOv5 알고리즘

YOLOv5는 You Only Look Once (YOLO) 시리즈의 최신 실시간 객체 탐지 알고리즘입니다. 그 속도와 정확도로 유명하며, 자원이 제한된 환경에서의 실시간 응용 프로그램에 특히 적합합니다. YOLOv5는 객체 분류와 위치 지정을 위한 단일 합성곱 신경망(CNN) 패스를 사용합니다. 멀티 스케일 예측과 Path Aggregation Network (PANet)을 통해 탐지 정확도를 향상시켰습니다. 또한, 개선된 앵커 메커니즘, 손실 함수 및 데이터 증강 전략을 도입하여 성능을 더욱 향상시켰습니다. YOLOv5의 이러한 특성들은 건설 안전 모니터링과 같이 빠르고 정확한 탐지가 요구되는 응용 프로그램에 매우 적합합니다.

IV. Evaluation & Analysis - 

V. Related Work (e.g., existing studies) - Tools, libraries, blogs, or any documentation that you have used to do this project. 

- (1)관련 논문：
A.	H. M. Rubaiyat et al., "Automatic Detection of Helmet Uses for Construction Safety," 2016 IEEE/WIC/ACM International Conference on Web Intelligence Workshops (WIW), Omaha, NE, USA, 2016, pp. 135-142, doi: 10.1109/WIW.2016.045.
B.	Li, Y., Wei, H., Han, Z., Huang, J., & Wang, W. (2020). Deep Learning-Based Safety Helmet Detection in Engineering Management Based on Convolutional Neural Networks. Advances in Civil Engineering, 2020, 9703560. https://doi.org/10.1155/2020/9703560

-(2)Libraries：
	https://github.com/ultralytics/yolov5；
	https://developer.nvidia.com/cuda-toolkit
 
-(3)IDE： https://www.jetbrains.com/zh-cn/pycharm/

VI. Conclusion: Discussion



