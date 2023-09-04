import cv2
import f_utils
import dlib
import numpy as np
from f_emotion_detection import predict_emotions


# instaciar detectores
frontal_face_detector    = dlib.get_frontal_face_detector()
emotion_detector         = predict_emotions()


def detect_liveness(im):
    
    gray = gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # face detection
    rectangles = frontal_face_detector(gray, 0)
    boxes_face = f_utils.convert_rectangles2array(rectangles,im)
    if len(boxes_face)!=0:
        # use only the biggest face
        areas = f_utils.get_areas(boxes_face)
        index = np.argmax(areas)
        rectangles = rectangles[index]
        boxes_face = [list(boxes_face[index])]

        # -------------------------------------- emotion_detection ---------------------------------------
        '''
        input:
            - imagen RGB
            - boxes_face: [[579, 170, 693, 284]]
        output:
            - status: "ok"
            - emotion: ['happy'] or ['neutral'] ...
            - box: [[579, 170, 693, 284]]
        '''
        _,emotion = emotion_detector.get_emotion(im,boxes_face)

    # -------------------------------------- output ---------------------------------------
    output = {
        'box_face_frontal': boxes_face,
        'emotion': emotion
    }
    return output

