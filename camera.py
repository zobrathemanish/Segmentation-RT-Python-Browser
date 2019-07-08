import cv2
import imutils
import time
import numpy as np

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture('cropvideo.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):

        CLASSES = open('enet-cityscapes/enet-classes.txt').read().strip().split("\n")

        # if a colors file was supplied, load it from disk

        COLORS = open('enet-cityscapes/enet-colors.txt').read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")
        
        net = cv2.dnn.readNet('enet-cityscapes/enet-model.net')

        (grabbed, frame) = self.video.read()

        frame = imutils.resize(frame, width=1000)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        output = net.forward()
        end = time.time()

        (numClasses, height, width) = output.shape[1:4]

        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        classMap = np.argmax(output[0], axis=0)

        mask = COLORS[classMap]

        # resize the mask such that its dimensions match the original size
        # of the input frame
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input frame with the mask
        # to form an output visualization
        output = ((0.15 * frame) + (0.7 * mask)).astype("uint8")


        ret, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()
