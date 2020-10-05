# virtual-try-on
의류 가상 피팅

# Tree
```
.
├── mrcnn/
│   ├── config.py
│   ├── model.py
│   ├── t_shirt.py
│   └── utils.py
├── vton/
│   └── utils.py       : 데이터 정제 유틸
├── data_cleaning.py   : 데이터 정제
└── train.py           : 티셔츠 모델 학습
```

# Skills
- python3
- Numpy
- Keras
- HDF5
- TensorFlow
- scikit-image
- Mask_RCNN

# How to use  
#### 데이터
데이터는 [여기](https://github.com/switchablenorms/DeepFashion2)에서 다운로드 받을 수 있으며, 해당 데이터를 최상위 경로에 아래와 같은 구조로 위치시킵니다.
```
.
├── data/
│   ├── train/
│   │   ├── annos/
│   │   └── image/
│   └── validation/
│       ├── annos/
│       └── image/
├── mrcnn/
│   ├── config.py
│   ├── model.py
│   ├── t_shirt.py
│   └── utils.py
├── vton/
│   └── utils.py
├── data_cleaning.py
└── train.py
```

#### Mask_RCNN 모델
[여기](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)에서 Mask_RCNN 모델을 다운로드 받아 아래와 같은 구조로 위치시킵니다.
```
.
├── data/
│   ├── train/
│   │   ├── annos/
│   │   └── image/
│   └── validation/
│       ├── annos/
│       └── image/
├── mrcnn/
│   ├── config.py
│   ├── mask_rcnn_coco.h5
│   ├── model.py
│   ├── t_shirt.py
│   └── utils.py
├── vton/
│   └── utils.py
├── data_cleaning.py
└── train.py
```

#### 데이터 전처리
티셔츠 데이터만 분류하기 위해 아래 명령어를 입력합니다.
- train data : `python data_cleaning.py --data train`
- validation data : `python data_cleaning.py --data validation`

#### 모델 학습
분류한 티셔츠 데이터를 이용하여 모델을 학습합니다.
`python train.py --mode train`

# Reference
- [쇼핑몰 의류 가상착용 서비스 블로그](https://mylifemystudy.tistory.com/category/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/AI%20School%20%3A%3A%20%EC%87%BC%ED%95%91%EB%AA%B0%20%EC%9D%98%EB%A5%98%20%EA%B0%80%EC%83%81%EC%B0%A9%EC%9A%A9%20%EC%84%9C%EB%B9%84%EC%8A%A4)
- [쇼핑몰 의류 가상착용 서비스 github](https://github.com/starbucksdolcelatte/FittingroomAnywhere)
- [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

<br>

---
  
<br>

#### Open Source License는 [이곳](NOTICE.md)에서 확인해주시고, 문의사항은 [Issue](https://github.com/IllIIIllll/virtual-try-on/issues) 페이지에 남겨주세요.
