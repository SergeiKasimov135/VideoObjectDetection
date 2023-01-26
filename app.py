import os
import detector as dt

def main() -> None:
    video_path = os.path.join("test_video", "students.mp4")
    config_path = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    model_path = os.path.join("model_data", "frozen_inference_graph.pb")
    classes_path = os.path.join("model_data", "coco.names")
    
    detector = dt.Detector(video_path=video_path, config_path=config_path,
                model_path=model_path, classes_path=classes_path)
    detector.video_processing()

if __name__ == "__main__":
    main()
