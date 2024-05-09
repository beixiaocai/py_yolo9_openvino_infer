from controllers.BaseHandler import BaseHandler
import json
import numpy as np
import base64
import cv2
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type
import time
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

        # Create random colors
        np.random.seed(42)  # Setting seed for reproducibility

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
        # t1 = time.time()
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
        # t2 = time.time()
        # print("predict spend %.4f" % (t2 - t1))
        return detections

device = "GPU"
print('device:', device)
CURRENT_DIR = os.path.dirname(__file__)
parent_dir = os.path.dirname(CURRENT_DIR)
model_path = os.path.join(parent_dir,"data/gelan-c.onnx")
print("AlgorithmHandler.py model_path=", model_path)
model = Yolov9(model_path=model_path)

class AlgorithmHandler(BaseHandler):
    async def post(self, *args, **kwargs):
        data = await self.do()
        self.response_json(data)

    async def do(self):
        request_params = self.request_post_params()

        # request_params_copy = request_params
        # del request_params_copy["image_base64"]
        # print(request_params.keys())

        happen = False
        happenScore = 0.0
        detects = []

        image_base64 = request_params.get("image_base64", None)  # 接收base64编码的图片并转换成cv2的图片格式
        if image_base64:
            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)

            # image = tj.decode(image_array)  # turbojpeg 解码
            image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)  # opencv 解码

            img_resized, dw, dh = model.resize_and_pad(image)
            results = model.predict(img_resized)


            if len(results) > 0:
                rx = image.shape[1] / (model.input_width - dw)
                ry = image.shape[0] / (model.input_height - dh)

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

                    detect = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "class_score": class_score,
                        "class_name": class_names[class_index]
                    }
                    # print(detect)

                    detects.append(detect)

        if len(detects) > 0:
            happen = True
            happenScore = 1.0

        res = {
            "code": 1000,
            "msg": "success",
            "result": {
                "happen": happen,
                "happenScore": happenScore,
                "detects": detects
            }
        }
        return res
