import cv2
import time
import numpy as np
import exception as ex

np.random.seed(20)


class Detector:
    def __init__(self: str, video_path: str, config_path: str,
                 model_path: str, classes_path: str) -> None:
        self.video_path = video_path
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        
        
        self.model = cv2.dnn_DetectionModel(model_path, config_path)
        self.model.setInputSize(320, 320)
        self.model.setInputScale(1.0/127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)
        
        self.read_classes()
        
    def read_classes(self) -> None:
        with open(self.classes_path, "r") as file:
            self.classes_list = file.read().splitlines()
        self.classes_list.insert(0, "__Background__")
        
        self.color_list = np.random.uniform(low=0, high=255,
                                            size=(len(self.classes_list), 3))

        print(self.classes_list)
        
    def video_processing(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened() == False:
            raise ex.CantOpenVideo

        fig, img = cap.read()
        while fig:
            class_label_IDs, confidences, bboxs = self.model.detect(img, 
                                                            confThreshold=0.4)
            
            bboxs = list(bboxs)
            
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            
            bbox_idx = cv2.dnn.NMSBoxes(bboxs, confidences, 
                                        score_threshold=0.5, nms_threshold=0.2)
            
            if len(bbox_idx) != 0:
                for i in range(len(bbox_idx)):
                    bbox = bboxs[np.squeeze(bbox_idx[i])]
                    class_confidence = confidences[np.squeeze(bbox_idx[i])]
                    class_label_ID = class_label_IDs[np.squeeze(bbox_idx[i])]
                    class_label = self.classes_list[class_label_ID]
                    class_color = [int(c) for c in self.color_list[class_label_ID]]

                    display_text = "{}:{:.2f}".format(class_label, class_confidence)
                    x, y, w, h = bbox

                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  color=class_color, thickness=1)
                    cv2.putText(img, display_text, (x, y-10), 
                                cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)
                    
            cv2.imshow("result", img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            
            fig, img = cap.read()

        cv2.destroyAllWindows()
