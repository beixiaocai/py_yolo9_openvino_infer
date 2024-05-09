import time
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2
import argparse
import os

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]


class Yolov9:
    def __init__(self, model_path, conf=0.2, nms=0.4):
        # Step 1. Initialize OpenVINO Runtime core
        core = ov.Core()
        # Step 2. Read a model
        ov_model = core.read_model(model_path)

        # Step 3. Inizialize Preprocessing for the model
        ppp = PrePostProcessor(ov_model)
        # Specify input image format
        ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
        #  Specify preprocess pipeline to input image without resizing
        ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
        # Specify model's input layout
        ppp.input().model().set_layout(Layout("NCHW"))
        #  Specify output results format
        ppp.output().tensor().set_element_type(Type.f32)
        # Embed above steps in the graph
        ov_model = ppp.build()

        self.compiled_model = core.compile_model(ov_model, "CPU")

        self.input_width = 640
        self.input_height = 640
        self.conf_thresh = conf
        self.nms_thresh = nms
        self.colors = []

        # Create random colors
        np.random.seed(42)  # Setting seed for reproducibility

        for i in range(len(class_names)):
            color = tuple(np.random.randint(100, 256, size=3))
            self.colors.append(color)

        self.infer_request = self.compiled_model.create_infer_request()

    def resize_and_pad(self, image):

        old_size = image.shape[:2]
        ratio = float(self.input_width / max(old_size))  # fix to accept also rectangular images
        new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = self.input_width - new_size[1]
        delta_h = self.input_height - new_size[0]

        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

        return new_im, delta_w, delta_h

    def predict(self, img):
        t1 = time.time()
        # Step 4. Create tensor from image
        input_tensor = np.expand_dims(img, 0)

        # Step 5. Create an infer request for model inference

        self.infer_request.infer({0: input_tensor})

        # Step 6. Retrieve inference results
        output = self.infer_request.get_output_tensor()
        detections = output.data[0].T

        # Step 7. Postprocessing including NMS
        boxes = []
        class_ids = []
        confidences = []
        for prediction in detections:
            classes_scores = prediction[4:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > self.conf_thresh):
                confidences.append(classes_scores[class_id])
                class_ids.append(class_id)
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)

        detections = []
        for i in indexes:
            j = i.item()
            detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})
        t2 = time.time()

        print("predict spend %.4f" % (t2 - t1))
        return detections

    def draw(self, img, results, dw, dh):
        rx = img.shape[1] / (self.input_width - dw)
        ry = img.shape[0] / (self.input_height - dh)

        for result in results:
            box = result["box"]
            class_index = result["class_index"]
            class_score = float("%.3f" % float(result["confidence"]))

            x1 = int(rx * box[0])
            y1 = int(ry * box[1])
            width = int(rx * box[2])
            height = int(ry * box[3])

            x2 = x1 + width
            y2 = y1 + height

            # Drawing detection box
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          tuple(map(int, self.colors[class_index])), 3)

            # Detection box text
            class_name = class_names[class_index] + ' ' + str(class_score)[:4]
            text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
            text_rect = (x1, y1 - 40, text_size[0] + 10, text_size[1] + 20)

            cv2.rectangle(img,
                          (int(text_rect[0]), int(text_rect[1])),
                          (int(text_rect[0] + text_rect[2]), int(text_rect[1] + text_rect[3])),
                          tuple(map(int, self.colors[class_index])), cv2.FILLED)

            cv2.putText(img, class_name, (int(x1 + 5), int(y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                        2, cv2.LINE_AA)


def process_image(image_path):
    img = cv2.imread(image_path)
    img_resized, dw, dh = model.resize_and_pad(img)
    results = model.predict(img_resized)
    model.draw(img, results, dw, dh)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            process_image(model, image_path)

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_resized, dw, dh = model.resize_and_pad(frame)
        results = model.predict(img_resized)
        model.draw(frame, results, dw, dh)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("test")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="data/gelan-c.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
        default='data/test.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.3,
        help="NMS threshould.",
    )
    args = parser.parse_args()

    model = Yolov9(args.model)

    if args.data_path.endswith('.jpg') or args.data_path.endswith('.png'):
        process_image(args.data_path)
    elif os.path.isdir(args.data_path):
        process_folder(args.data_path)
    elif args.data_path.endswith('.mp4'):
        process_video(args.data_path)
    else:
        print("unsupported file format")
