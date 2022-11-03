# K2T_CN
中文受限制文本生成任务。该项目继承自[gagan3012](https://github.com/gagan3012/keytotext)的keytotext项目，但是采用了中文T5预训练模型。
## 使用方法

### 1. 安装依赖包

``` python
  pip install keytotext --upgrade
  # 如果在windows环境下用pip install git-lfs
  sudo apt-get install git-lfs
  pip install pytorch-lightning==1.6.5
  pip install opencv-python
```

### 2. 关键词造句

 [model_v4.0](https://drive.google.com/drive/folders/1Ik-o_sZ5TUZEvU94fPmCC0JEGuSrPTo1?usp=sharing)

``` python
  from keytotext import trainer
  
  model = trainer()
  model.load_model("./model_v4.0", use_gpu = True)
  
  # 关键词可以是任意个数，use_gpu必须与上面一致
  keywords = ["天空", "山脉", "海洋"]
  print(model.predict(keywords, use_gpu = True))
```

## 训练方法

### 1. 安装依赖包

``` python
  pip install keytotext --upgrade
  sudo apt-get install git-lfs
  pip install pytorch-lightning==1.6.5
  pip install -U jiagu
  git clone https://github.com/ownthink/Jiagu
  cd Jiagu
  python3 setup.py install
  cd ../
```

### 2. 准备数据源（生成DataFrame格式）

<img width="1209" alt="截屏2022-10-17 15 57 52" src="https://user-images.githubusercontent.com/47048401/196121153-b8f1e95d-20fe-4256-b5ef-a06ac804db59.png">

### 3. 训练

  ``` python
  from keytotext import trainer
  
  model = trainer()
  
  '''
      model_name从HuggingFace的Model中选取一个，如果没有被HuggingFace官方收录的，则要带上模型作者的用户名，即：用户名/模型名
      huggingface模型地址：https://huggingface.co/models
  '''
  model.from_pretrained(model_name="IDEA-CCNL/Randeng-T5-77M")
  model.train(train_df=data[:1000], test_df=data[:500], batch_size=2, max_epochs=2,use_gpu=True)
  model.save_model()
```
  
  
