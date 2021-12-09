import cv2
import time
import numpy as np

import pickle
import tensorflow.lite as tflite

WINDOW_NAME = "webcam window"
MODEL_INPUT_SIZE = (256, 256)
MODEL_PATH = "pose_thunder_float16.tflite"


LABEL = "X"
DATA_PATH = "data.csv"

# global data, output, data_counter

data_counter = 0
data = {"data":[], "label":[]}



class PoseEstimator:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array([image]))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])


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

def add_data():
    new_data = []
    for point in output[5:]:
        if(point[0] > 1 or point[1] > 1):
            print("!!!!!!!!!--- INVALID POINT ---!!!!!!!!!")
            return
        else:
            new_data.append([point[1], point[0]])
    data["data"].append(new_data)
    data["label"].append(LABEL)

    global data_counter
    data_counter += 1

    print(new_data)
    print(data_counter)





def onMouse(event, x, y, _, __):
    if event == cv2.EVENT_LBUTTONDOWN:
        add_data()



if __name__ == '__main__':
    # data_frame =  pd.DataFrame({"data": [], "label":[]})
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, onMouse)
    estimator = PoseEstimator(MODEL_PATH)
    camera = cv2.VideoCapture(0)
    frame_rate = 15
    prev = 0

    while True:
        time_elapsed = time.time() - prev
        _, frame = camera.read()
        image = frame
        if(time_elapsed > 1./frame_rate):
            prev = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, MODEL_INPUT_SIZE)

            output_data = estimator.run(image)

            global output
            output = np.squeeze(output_data)
            drawLines(image, output_data)
            cv2.imshow(
                WINDOW_NAME,
                cv2.resize(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                        ),
                    (1200, 800)
                    )
                )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # pd.DataFrame(data).to_csv(f"{LABEL}.csv", index = None, header=True)
            with open(f'{LABEL}.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                # The following example reads the resulting pickled data.
            break