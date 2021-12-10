import cv2
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


def drawLines(image, output_data):
    points = {}
    width, height, _ = image.shape
    for idx, point in enumerate(np.squeeze(output_data)):
        y = point[0] * height
        x = point[1] * width
        confidence = point[1]
        if(x <= width and y <= height):
            points[idx] = (int(x), int(y))

    drawLine(image, points, 8, 10)
    drawLine(image, points, 6, 8)
    drawLine(image, points, 5, 6)
    drawLine(image, points, 5, 7)
    drawLine(image, points, 7, 9)
    drawLine(image, points, 5, 11)
    drawLine(image, points, 6, 12)
    drawLine(image, points, 11, 12)
    drawLine(image, points, 11, 13)
    drawLine(image, points, 13, 15)
    drawLine(image, points, 12, 14)
    drawLine(image, points, 14, 16)


def drawLine(image, points, idx1, idx2):
    if(idx1 in points and idx2 in points):
        point1 = points[idx1]
        point2 = points[idx2]
        cv2.line(image, point1, point2, thickness=2, color=(0, 255, 0))


while True:
    _, frame = camera.read()


    frame_counter += 1
    image = frame
    if(True):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        interpreter.set_tensor(input_details[0]['index'], np.array([image]))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        classifier_input = np.squeeze(output_data.copy())[5:, 0:2]
        temp = np.copy(classifier_input[:, 0])
        classifier_input[:, 0] = classifier_input[:, 1]
        classifier_input[:, 1] = temp
        classifier.set_tensor(cls_input_details[0]["index"], [classifier_input.ravel()])

        classifier.invoke()
        output = classifier.get_tensor(cls_output_details[0]["index"])

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


        drawLines(image, output_data)

    cv2.imshow("webcam", cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (800, 800)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break