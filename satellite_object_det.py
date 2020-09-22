#Install Packages
#!pip install --upgrade tensorflow
!pip install opencv-python
!pip install keras==2.3.1
!pip install tensorflow-gpu==1.15.0
!pip install imageai --upgrade

#The below function is used to change the current working directory
import os
print('Previous current working directory: '+os.getcwd()+'\n')
from google.colab import drive  
drive.mount('/gdrive')
#Below you should write the path to the desired directory
%cd /gdrive/My Drive/DLSC
print('\nNew current working directory: '+os.getcwd())

# Training

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Dataset2")
trainer.setTrainConfig(object_names_array=["Vehicle"], batch_size=4, num_experiments=20, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

# Evaluation

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Dataset2")
metrics = trainer.evaluateModel(model_path="Dataset2/models", json_path="Dataset2/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)


# Detection

from imageai.Detection.Custom import CustomObjectDetection


def deet(fold,best):
  detector = CustomObjectDetection()
  detector.setModelTypeAsYOLOv3()
  detector.setModelPath("Dataset{}/models/{}.h5".format(fold,best))
  detector.setJsonPath("Dataset{}/json/detection_config.json".format(fold))
  detector.loadModel()
  for ikl in [1,2,3,4,5,6]:
    detections = detector.detectObjectsFromImage("Dataset{}/test/T{}.jpg".format(fold,ikl), output_image_path="Dataset{}/test/T{}_detected.jpg".format(fold,ikl),nms_treshold=0.1, minimum_percentage_probability=30)
    print("T{}".format(ikl))
    for detection in detections:
      print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
      
      
      
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Dataset2")
metrics = trainer.evaluateModel(model_path="Dataset2/models", json_path="Dataset2/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)


deet(2,'detection_model-ex-018--loss-0035.111')
