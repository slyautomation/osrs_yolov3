# osrs_yolov3
Yolov3 Object Detection In OSRS using Python code, Detecting Cows - Botting

name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7085 (Average: 12 FPS)

![ezgif-7-7bcf90e20cc2](https://user-images.githubusercontent.com/81003470/132772257-80ac3835-7b7b-4f30-9ba5-91f7999506b5.gif)

 ![image](https://user-images.githubusercontent.com/81003470/116421155-ef104300-a881-11eb-930d-56b4b93511fd.png)


Quick Start

Download YOLOv3 weights from YOLO website. Run in Pycharm Terminal: 

![image](https://user-images.githubusercontent.com/81003470/111890820-5ea04080-8a41-11eb-8fea-daf0a551bf07.png)

https://pjreddie.com/media/files/yolov3.weights ----- save this in 'model_data' directory

Convert the Darknet YOLO model to a Keras model.

type in terminal: convert.py -w model_data/yolov3.cfg model_data/yolov3.weights model_data/yolov3.h5

type pip install -r requirements

goto Google drive for large files and specifically the osrs cow and goblin weighted file: https://drive.google.com/folderview?id=1P6GlRSMuuaSPfD2IUA7grLTu4nEgwN8D

and see full video tutorial: https://youtu.be/X3snnwzJfEw

Check if your gpu will work: https://developer.nvidia.com/cuda-gpus and use the cuda for your model and the latest cudnn for the cuda version.

pycharm = https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC

cuda 10.0 = https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10

cudnn = https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-10

labelImg = https://tzutalin.github.io/labelImg/

tesseract-ocr = https://sourceforge.net/projects/tesseract-ocr-alt/files/tesseract-ocr-setup-3.02.02.exe/download

Credit to: https://github.com/pythonlessons/YOLOv3-object-detection-tutorial

## Trouble Shooting:

change tensorflow_backend.py line 95 to tf.compat.v1.reset_default_graph()
change tensorflow_backend.py line 98 to phase = tf.compat.v1.placeholder_with_default(False,
change tensorflow_backend.py line 102 to _GRAPH_LEARNING_PHASES[tf.compat.v1.get_default_graph()] = phase
