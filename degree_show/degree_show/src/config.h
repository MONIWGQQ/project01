#include "def.h"
#include <string>


using namespace std;
class Config{
public:
    Config(string yolo_face_model_path, string hand_detection_model_path, string image_path){
        this->yolo_face_model_path = yolo_face_model_path;
        this->hand_detection_model_path = hand_detection_model_path;
        this->image_path = image_path;
        this->face_confThreshold = 0.45;
        this->face_nmsThreshold = 0.5;
        this->hand_confThreshold = 0.8; // 为了防止左右手误检，这里设置为高阈值0.8
    }
    Config(){
        this->yolo_face_model_path = "/Users/moniwang/of_v0.12.0_osx_release/apps/degree_show2/degree_show/bin/weights/yolov7-w6-face.onnx";
        this->hand_detection_model_path = "/Users/moniwang/of_v0.12.0_osx_release/apps/degree_show2/degree_show/bin/weights/yolo_model_hands.onnx";
        this->image_path = "/Users/moniwang/of_v0.12.0_osx_release/apps/degree_show2/degree_show/bin/data/landscape.jpg";
        this->face_confThreshold = 0.45;
        this->face_nmsThreshold = 0.5;
        this->hand_confThreshold = 0.8; // 为了防止左右手误检，这里设置为高阈值0.8
    }
    Net_config get_face_config(){
        Net_config config;
        config.confThreshold = this->face_confThreshold;
        config.nmsThreshold = this->face_nmsThreshold;
        config.modelpath = this->yolo_face_model_path;
        return config;
    }
    Net_config get_hand_config(){
        Net_config config;
        config.confThreshold = this->hand_confThreshold;
        config.nmsThreshold = this->hand_confThreshold;
        config.modelpath = this->hand_detection_model_path;
        return config;
    }

    string get_image_path(){
        return this->image_path;
    }
private:
    string yolo_face_model_path;
    float face_confThreshold;
    float face_nmsThreshold;
    string hand_detection_model_path;
    float hand_confThreshold;
    string image_path;
};
