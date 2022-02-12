from __future__ import division
import cv2
import time
import sys

conf_threshold = 70 #percentage

def show_text(text, frame, where, color):
    cv2.putText(frame, text, (int(where[0]), int(where[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def detect_face_OpenCV_DNN(net, frame):
    faces_frame = frame.copy()
    frame_height = faces_frame.shape[0]
    frame_width = faces_frame.shape[1]

    blob = cv2.dnn.blobFromImage(faces_frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    detections = detections[0, 0]
    for det in detections:
        confidence = det[2]
        if confidence > conf_threshold/100.:
            x1 = int(det[3] * frame_width)
            y1 = int(det[4] * frame_height)
            x2 = int(det[5] * frame_width)
            y2 = int(det[6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(faces_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
            show_text("{:.0f}".format(confidence*100), faces_frame, (x1, y1 - 10), (0, 0, 255))
    return faces_frame, face_boxes

if __name__ == "__main__" :

    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    source = 0 #from the webcam
    #source = "Faces from around the world.avi"
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    has_frame, frame = cap.read()

    vid_writer_in  = cv2.VideoWriter('input.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1], frame.shape[0]))
    vid_writer_out = cv2.VideoWriter('output-dnn.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    while (1):
        has_frame, frame = cap.read()
        if not has_frame:
            break
        vid_writer_in.write(frame)
        frame_count += 1

        t = time.time()
        faces_frame, face_boxes = detect_face_OpenCV_DNN(net,frame)
        duration = time.time() - t

        show_text("Face detection FPS: {:.2f}".format(1 / duration), faces_frame, (10, 50), (0, 0, 255))
        show_text("Frame number: {:.0f}".format(frame_count), faces_frame, (10, 100), (0, 0, 255))
        cv2.imshow("Face Detection", faces_frame)

        if True:
            vid_writer_out.write(faces_frame)

        k = cv2.waitKey(10)
        if k == ord(' '): #pause: useful in the developing phase
            while cv2.waitKey(10) == -1:
                pass
        if k == 27:
            break

    cv2.destroyAllWindows()
    vid_writer_in.release()
    vid_writer_out.release()
