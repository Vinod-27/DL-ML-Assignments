# Car_Person:::[Documentation]

#1) Model Name :- YOLOv5 
   
#2) Links to dataset and framework :- 
   https://evp-ml-data.s3.us-east-2.amazonaws.com/mlinterview/openimages-personcar/trainval.tar.gz ,
   https://pytorch.org/docs/stable/index.html
   
#3) About the Model :-
   YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, 
   and includes simple functionality for Test Time Augmentation (TTA), model ensembling, 
   hyperparameter evolution, and export to ONNX, CoreML and TFLite. Model.
   
#4) Primary Analysis :- 
   The YOLO network consists of three main pieces.
    1) Backbone - A convolutional neural network that aggregates and forms image features at different granularities.
    2) Neck - A series of layers to mix and combine image features to pass them forward to prediction.
    3) Head - Consumes features from the neck and takes box and class prediction steps.
  
    YOLO model is fast and blazingly fast than the other models because In a YOLOv5 Colab notebook, running a Tesla P100, we saw inference times up to 0.007 seconds per image,         meaning 140 frames per second (FPS)! By contrast, YOLOv4 achieved 50 FPS after having been converted to the same Ultralytics PyTorch library.
    
#5) Assumptions :-
   Assumptions of yolov5 model requires labels in text format,it cannot take any other files except text file,but in this dataset the annotation are given in 
   json  format.YOLOv5 is detect very fast in small objects and YOLO object detection is more than 1000x faster than R-CNN and 100x faster than Fast R-CNN. YOLO model is based      on  regression,There is no need for us to train all the images for training purpose,we can give 25% images for predictions.Always try to best annotations for labeling,
   because model will not be able to detect objects correctly when you did not annotation properly.  

#6) Inference :- 
   What we need is a Deep Learning model that works on our testing data. So for test model on testing data  will have to use the “model = torch.hub.load('D:\DL
   Assignment\yolov5','custom', r'D:\DL Assignment\models\best.pt',source='local') ” script present at the location “main.py”. We are giving test images for prediction
   " img = r'D:\DL Assignment\test_images\15.jpg' " and the predicted images are located in predicted_images folder.
   main.py: python file for inferencing.
   test_images:the path of testing data or testing image file.
   img: image size which must be the same as training image size.
   Running the main.py file would create a folder “runs/detect/exp” which would have all the outcome images with detection.
   
 #7) Approach :- 
    The given dataset contains annotation and images,but the annotation are in json format which is not supported by yolov5 model,I have taken all images for
    annotate ,i used pylabel this will give annotations as well as we can see the annotate images,I applied pylabel importer on specific path of annoatation json file and
    path to images which is required for labeling.When i run this script "dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="trainval")"
    it will give img_widh,img_height,bbox ,segmentation ,cat_name and so on. The PyLabel exporter will export all of the annotations in the dataframe to the desired target
    format. Yolo creates one text file for each image in the dataset by this line "dataset.export.ExportToYoloV5()[0]".Finally i got annotation of each image.
    
    When come to the part of model training ,i have cloned a yolov5 model with github link "!git clone https://github.com/ultralytics/yolov5" then i created a folder for
    training images ,i had given images to images folder(train&val) labels to labels folder(train&val),dataset.yaml file consist two paths for training and val which is required     from images and labels folder.No of classes nc=2 because we have to detect two categories which was car and person. "! pip install -qr requirements.txt" this script will
    automatically install all required files for this project. 
    "! python train.py --img 2239 --batch 4 --epochs 8 --data /content/dataset.yaml --weights yolov5s.pt" I had given batched 4 and epochs 8 with path dataset.yaml file
    the above script has successfully executed without any error and results are saved in runs/train/exp , Finally we got another file i.e best.pt with this file we can test
    images for predictions.
    
    So for test model on testing data  will have to use the “model = torch.hub.load('D:\DL Assignment\yolov5','custom', r'D:\DL Assignment\models\best.pt',source='local') ” 
    script present at the location “main.py”. We are giving test images for prediction 
    " img = r'D:\DL Assignment\test_images\15.jpg' " and the predicted images are located in predicted_images folder.
    main.py: python file for inferencing.
    test_images:the path of testing data or testing image file.
    img: image size which must be the same as training image size.
    best.pt : it is located in models folder and outide file 
    Running the main.py file would create a folder “runs/detect/exp” which would have all the outcome images with detection.
    We can see the predicted imaged of object detection "in run/detect/exp" folder.
    
#8) False positives :- 
   In some image predictions a normal truck is detected as a car becuase truck tires and size of truck ,wheels are similar to car.Some times a baner images
   are detecting like a person,so this things i had observed in false positives.
  
#9) Metrics :- 
    Model Summary: 213 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs

#10) Conclusion :-
     The trained model will given accurate results,we can see in predicted images folder.I had learned lots of things with this project and also i faced lots of
     errors while running Train_Test.py file.The main thing i got advantage from pylabel package for image annotations.From this project we can easily detect persons and car in
     live video.
    
