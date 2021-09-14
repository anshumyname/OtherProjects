import cv2
from os.path import dirname, join
import os 
print(os.getcwd())
protoPath = join(os.getcwd(), "pose_deploy_linevec_faster_4_stages.prototxt")
modelPath = join(os.getcwd(), "pose_deploy_linevec.prototxt")
print(protoPath)

# protoFile = "C:/Users/sriva/Documents/WebD/human_video/test/OpenPose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "C:/Users/sriva/Documents/WebD/human_video/test/OpenPose/pose/empi/pose_deploy_linevec.prototxt"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)