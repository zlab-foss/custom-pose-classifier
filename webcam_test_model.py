import cv2
import time
import numpy as np
import tensorflow.lite as tflite



font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20,160)
fontScale              = .9
fontColor              = (255 ,0,100)
thickness              = 2
lineType               = 1



CLASSIFIER_LABELS = ("A", "T", "W", "X", "Y")

interpreter = tflite.Interpreter(model_path="pose_thunder_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

classifier = tflite.Interpreter(model_path="classify_model.tflite")
classifier.allocate_tensors()
cls_input_details = classifier.get_input_details()
cls_output_details = classifier.get_output_details()
cls_input_shape = input_details[0]['shape']

camera = cv2.VideoCapture(0)
frame_counter = 0
while True:
    _, frame = camera.read()


    frame_counter += 1
    image = frame
    # if(frame_counter%30 == 0):
    if(True):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        interpreter.set_tensor(input_details[0]['index'], np.array([image]))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        start_time = time.time()
        classifier_input = np.squeeze(output_data)[5:, 0:2]
        classifier.set_tensor(cls_input_details[0]["index"], [classifier_input])
        classifier.invoke()
        output = classifier.get_tensor(cls_output_details[0]["index"])
        print(round(time.time() - start_time, 5), "s")

        output = np.squeeze(output)
        arg = np.argmax(output)
        pred = CLASSIFIER_LABELS[arg]
        conf = int(output[arg]*100)
        cv2.putText(image,f"{pred}:{conf}%",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        width, height, _ = image.shape
        for point in np.squeeze(output_data):
            y = point[0]
            x = point[1]
            if(x <= 1 and y <= 1):
                image = cv2.circle(image, (int(x*width), int(y*height)), radius=1, color=(0, 0, 255), thickness=1)

        print(40*"-")



    cv2.imshow("webcam", cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (800, 800)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break