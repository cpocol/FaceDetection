from __future__ import division
import cv2
import time
import sys

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            label = "{:.0f}".format(confidence*100)
            cv2.putText(frameOpencvDnn, label, (int(x1), int(y1-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frameOpencvDnn, bboxes

if __name__ == "__main__" :

    print(sys.version)

    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    source = "Faces from around the world.avi"
    #source = 0 #from the webcam

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    vid_writerOut = cv2.VideoWriter('output-dnn.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1], frame.shape[0]))
    vid_writerIn  = cv2.VideoWriter('input.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    tt_opencvDnn = 0
    while (1):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        vid_writerIn.write(frame)
        frame_count += 1

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        label = "Frame# : {:.0f}".format(frame_count)
        cv2.putText(outOpencvDnn, label, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection", outOpencvDnn)

        #if frame_count > 500:
        if True:
            vid_writerOut.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(10)
        if k == 32: #pause: useful in the developing phase
            while cv2.waitKey(10) == -1:
                pass
        if k == 27:
            break
        
        #time.sleep(0.300)

    cv2.destroyAllWindows()
    vid_writerOut.release()
    vid_writerIn.release()

