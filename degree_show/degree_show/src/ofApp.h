#pragma once


#include "def.h"
#include "config.h"
#include "ofMain.h"
#include "ofxCv.h"
#include "ofVideoGrabber.h"
#include <onnxruntime_cxx_api.h>
// #include "lite/lite.h"
#include "ofxOpenCv.h"


*/
using namespace ofxCv;
using namespace cv;
// #define _USE_LIVE_VIDEO
// extern struct BoxInfo;
class YOLOV7_face;
class HandDetection;
class ofApp : public ofBaseApp{

    public:
        void setup() override;
        void update() override;
        void draw() override;
        void exit() override;

        void keyPressed(int key) override;
        void keyReleased(int key) override;
        void mouseMoved(int x, int y ) override;
        void mouseDragged(int x, int y, int button) override;
        void mousePressed(int x, int y, int button) override;
        void mouseReleased(int x, int y, int button) override;
        void mouseScrolled(int x, int y, float scrollX, float scrollY) override;
        void mouseEntered(int x, int y) override;
        void mouseExited(int x, int y) override;
        void windowResized(int w, int h) override;
        void dragEvent(ofDragInfo dragInfo) override;
        void gotMessage(ofMessage msg) override;
    
    ofVideoGrabber cam; // Video grabber to access the webcam
    // CascadeClassifier classifier; // OpenCV face classifier
    
    vector<cv::Rect> faces; 
    float rescale;
    
      ofImage image;
  
      ofMesh mesh;
      ofMesh meshCopy;
  
      ofColor c;
      ofEasyCam easyCam;
      ofColor rotateC;

      Config config; // Configuration file object
      bool using_video; // for testing video
      ofVideoPlayer player; // for testing video
      bool image_show;  // Whether to show the camera detection window in the 3D image window.
      YOLOV7_face *net; // face and keypoint detection mode
      HandDetection *hand_net; // Hand detection model
      string kWinName; // Name of the camera window.
      bool isInit;  // Whether to get the face detection result for the first time.
      BoxInfo lastBoxes; // The face detection result of the last frame.
      float ratio_thresh; // Threshold for determining if a head movement has occurred
      bool show_detection_windows; // Whether to show the camera detection window.
      float scaleFactor;   // the scale factor compared to the initial image
      char rotateDirection; // rotate direction around x, y, and z axes
      bool clockwise;  // Rotate clockwise or counterclockwise.
      // int detectimes;
      float lastActionTime; // last header detection time
      float actionDelay; // set a delay time threshold to prevent head motion detection
      uint frameCount; // Since two models are used, the hand is detected in the odd frames and the face in the even frames.
      bool pauseFlag; //  pause flag
      // int handDetectTimes; //  the number of times the hand is detected, which can be used to enhance the stability of the detection
      int orbiting; // 0 no head detected 1 head detected and zoomed in 2 head detected and zoomed out
      
      bool mouseDisplacement;
      float startOrbitTime;
  
      // These variables will let us store the polar coordinates of each vertex
      vector<float> distances;
      vector<float> angles;
      ofVec3f meshCentroid;
};
// ofApp::KwinName = "Deep learning object detection in ONNXRuntime";
