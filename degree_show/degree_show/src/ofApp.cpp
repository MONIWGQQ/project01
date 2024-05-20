#include <iostream>
#include <stdio.h>
// #include <direct.h>
#include <unistd.h>
#include "ofApp.h"
#include "ofxCv.h"
#include "ofVideoGrabber.h"
#include "ofYolo.h"
#include "ofhand.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace glm;
// using namespace ofxCv;
using namespace cv;
using namespace std;
//--------------------------------------------------------------
void ofApp::setup(){
    kWinName = "Deep learning object detection in ONNXRuntime";
//    ofSetupOpenGL(800,800,OF_WINDOW);
    cam.setup(640, 480); // Setup the video grabber
    // char *buffer;
	// //also use buffer as an output parameter
	// if((buffer = getcwd(NULL, 0)) == NULL)
	// {
	// 	perror("getcwd error");
	// }
	// else
	// {
    //     printf("Current Working Directory %s\n", buffer);
	// 	free(buffer);
	// }
    const int MAXPATH=250;
    char buffer[MAXPATH];
    getcwd(buffer, MAXPATH);
    printf("The current directory is: %s", buffer);

    //加载模型配置
    Config config;
    net = get_yolo_face_network(config.get_face_config());
    hand_net = get_hand_network(config.get_hand_config());

    image.load(config.get_image_path());
    image.resize(100, 100);
    mesh.setMode(OF_PRIMITIVE_LINES);
    mesh.enableColors();
    mesh.enableIndices();
    ofSetFrameRate(2500);
    
    float intensityThreshold =60.0;
    int w = image.getWidth();
    int h = image.getHeight();
    for(int x=0; x<w; ++x){
        for(int y=0; y<h; ++y){
            ofColor c= image.getColor(x,y);
        float intensity = c.getLightness();
        if(intensity >= intensityThreshold){
            float saturation = c.getSaturation();
                float z = ofMap(saturation, 0, 255, -100, 100);
                ofVec3f pos(x*6, y*6, z);
                mesh.addVertex(pos);
                mesh.addColor(c);
            }
        }
    }
//    cout << mesh.getNumVertices() << endl;
    float connectionDistance = 30;
    int numVerts = mesh.getNumVertices();
    for (int a=0; a<numVerts; ++a) {
        ofVec3f verta = mesh.getVertex(a);
        for (int b=a+1; b<numVerts; ++b) {
            ofVec3f vertb = mesh.getVertex(b);
            float distance = verta.distance(vertb);
            if (distance <= connectionDistance) {
                mesh.addIndex(a);
                mesh.addIndex(b);
            }
        }
    }
    meshCentroid = mesh.getCentroid();
    for (int i=0; i<numVerts; ++i) {
        ofVec3f vert = mesh.getVertex(i);
        float distance = vert.distance(meshCentroid);
        float angle = atan2(vert.y-meshCentroid.y, vert.x-meshCentroid.x);
        distances.push_back(distance);
        angles.push_back(angle);

        // Set the z-value based on the distance to the centroid
        vert.z = distance;  // or any function of distance
        mesh.setVertex(i, vert);
    }
    orbiting = 0;
    startOrbitTime = 0.0;
    meshCopy = mesh; // Store a copy of the mesh, so that we can reload the original state
    mouseDisplacement = false;
    // just for testing
    using_video = false;
    if(using_video){
        player.load("test2.avi");
        player.play();
    }


    image_show = false;
    isInit = false;
    ratio_thresh = 30.0;
    show_detection_windows = true;
    scaleFactor = 1.0f;
    rotateDirection = 'z';
    clockwise = true;
    // detectimes = 0;
    actionDelay = 0.55f;
    frameCount = 0;
    pauseFlag = false;
    // handDetectTimes = 0;
    // lastActionTime = ofGe
}

float distance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

std::string detectMovement(BoxInfo& face1, BoxInfo& face2) {
    float dx = face2.kpt3.pt.x - face1.kpt3.pt.x;
    float dy = face2.kpt3.pt.y - face1.kpt3.pt.y;

    // float dist1 = distance(face1.kpt1.pt, face1.kpt5.pt);
    // float dist2 = distance(face2.kpt1.pt, face2.kpt5.pt);
    float area1 = face1.calculateQuadrilateralArea();
    float area2 = face2.calculateQuadrilateralArea();
    // 判断靠近或远离的阈值
    float threshold = 1500.0f;

    // if (std::abs(dist2 - dist1) > threshold * std::abs(dx) && std::abs(dist2 - dist1) > threshold * std::abs(dy)) {
    // printf("Area1: %f, Area2: %f  diff %f\n", area1, area2, fabs(area1 - area2));
    if(fabs(area1 - area2)>threshold){
        if (area1 < area2) {
            return "Further";
        } else {
            return "Closer";
        }
    } else if (std::abs(dx) > std::abs(dy)) {
        if (dx > 0) {
            return "Right";
        } else {
            return "Left";
        }
    } else {
        if (dy > 0) {
            return "Up";
        } else {
            return "Down";
        }
    }
}

void rotateMesh(ofMesh& mesh, const ofVec3f& meshCentroid, const std::vector<float>& distances, const std::vector<float>& angles, float scaleFactor, float startOrbitTime, char axis, bool clockwise) {
    int numVerts = mesh.getNumVertices();
    for (int i=0; i<numVerts; ++i) {
        ofVec3f vert = mesh.getVertex(i);
        float distance = distances[i] * scaleFactor;
        float angle = angles[i];
        float elapsedTime = ofGetElapsedTimef() - startOrbitTime;
        float speed = ofMap(distance, 0, 200, 1, 0.25, true);
        float rotatedAngle = elapsedTime * speed + angle;

        // If the rotation is counterclockwise, subtract the angle from 360
        if (!clockwise) {
            rotatedAngle = 360 - rotatedAngle;
        }

        switch(axis) {
            case 'x':
                vert.y = distance * cos(rotatedAngle) + meshCentroid.y;
                vert.z = distance * sin(rotatedAngle) + meshCentroid.z;
                break;
            case 'y':
                vert.x = distance * cos(rotatedAngle) + meshCentroid.x;
                vert.z = distance * sin(rotatedAngle) + meshCentroid.z;
                break;
            case 'z':
                vert.x = distance * cos(rotatedAngle) + meshCentroid.x;
                vert.y = distance * sin(rotatedAngle) + meshCentroid.y;
                break;
            default:
                std::cout << "Invalid axis: " << axis << std::endl;
                return;
        }

        mesh.setVertex(i, vert);
    }
}

//--------------------------------------------------------------
void ofApp::update(){
    if(!using_video){
        cam.update(); // update video grabber
        // cout<<"update"<<endl;
        // cv::Mat camMat;
        // cap>>camMat;
        if(cam.isFrameNew()) {
            frameCount ++;
            ofPixels & pixels = cam.getPixels();
            cv::Mat camMat;
            camMat = ofxCv::toCv(pixels); // Initialize camMat with the camera pixels
            // printf("camMat size %d %d\n", camMat.cols, camMat.rows);
            cvtColor(camMat, camMat, COLOR_RGB2BGR);
            // cv::Mat copyMat = camMat.clone();
            vector<BoxInfo> generate_boxes;
            // vector<BoxInfo_Hand> faceboxes;
            if(frameCount%2==0 && !pauseFlag){
                net->detect(camMat, generate_boxes);
            }else{
                string res = hand_net->detect(camMat);
                if(res=="left"){
                    printf("start\n");
                    pauseFlag = false;
                }
                else if(res=="right"){
                    printf("stop\n");
                    pauseFlag = true;
                }
                else if(res=="both") printf("ambigious\n");
            }
            if(pauseFlag){
                if(show_detection_windows){
                    imshow(kWinName, camMat);
                }
                return;
            }
            size_t index=0;
            if(generate_boxes.size()>0){
                if(!isInit){
                    isInit = true;
                    lastBoxes = generate_boxes[index];
                    lastActionTime = ofGetElapsedTimef();
                }else{
                    // Sort by confidence level
                    for (size_t n = 1; n < generate_boxes.size(); n++){
                        if(generate_boxes[n].score > generate_boxes[index].score){
                            index = n;
                        }
                    }
                    float ratio = distance(generate_boxes[index].kpt3.pt, lastBoxes.kpt3.pt);

                    if(ratio > ratio_thresh){
                        std::string direc = detectMovement(generate_boxes[index], lastBoxes);
                        
                        // If the current time minus the time of the last action is greater than the delay, then start detecting a new action
                        if (ofGetElapsedTimef() - lastActionTime > actionDelay || (direc == "Closer" || direc == "Further")) {
                            if(direc == "Right"){
                                cout<<"Right"<<endl;
                                orbiting = 1;
                                rotateDirection = 'y';
                                clockwise = true;
                            }
                            else if(direc == "Left"){
                                cout<<"Left"<<endl;
                                orbiting = 2;
                                rotateDirection = 'y';
                                clockwise = false;
                            }
                            else if(direc == "Up"){
                                cout<<"Up"<<endl;
                                orbiting = 1;
                                rotateDirection = 'x';
                                clockwise = true;
                            }
                            else if(direc == "Down"){
                                cout<<"Down"<<endl;
                                orbiting = 2;
                                rotateDirection = 'x';
                                clockwise = false;
                            }
                            else if(direc == "Closer"){
                                cout<<"Closer"<<endl;
                                orbiting = 1;
                                rotateDirection = 'z';
                                clockwise = true;
                            }
                            else if(direc == "Further"){
                                cout<<"Further"<<endl;
                                orbiting = 2;
                                rotateDirection = 'z';
                                clockwise = false;
                            }
                            
                            // document the current time
                            lastActionTime = ofGetElapsedTimef();
                        }else{
                            // If the current time minus the time of the last action is less than the delay, then the action is not executed
                            orbiting = 0;
                        }
                        lastBoxes = generate_boxes[index];
                        // detectimes++;
                        
                    }else{
                        orbiting = 0;
                    }
                }
                
            }else{
                // 保持原状
                orbiting = 0;
            }
            if(show_detection_windows){
                imshow(kWinName, camMat);
            }
        }
    }else{
        // This is just for test just to see if the video is playing
        player.update();
        // sleep(1);
        if(player.isFrameNew()){
            ofPixels & pixels = player.getPixels();
            cv::Mat camMat;
            camMat = ofxCv::toCv(pixels); // Initialize camMat with the camera pixels
            // cv::Mat graySmallMat;
            // rescale = 0.5; // rescale factor for performance
            // cv::resize(camMat, graySmallMat, cv::Size(), rescale, rescale);
            // faces.clear();
            // cv::Size minSize_;
            // minSize_.width = 15;
            // minSize_.height = 15;
            // classifier.detectMultiScale(graySmallMat, faces, 1.2, 1, 0, minSize_);
            // if(faces.size()>0){
            //     orbiting = 1;
            //     std::cout<<"Detect "<<faces.size()<<" Face(s) Position is:"<<std::endl;
            // }else{
            //     orbiting = 0;
            // }
            // for(cv::Rect & face : faces) {
            //     face.x = face.x/rescale;
            //     face.y = face.y/rescale;
            //     face.width /= rescale;
            //     face.height /= rescale;
            //     std::cout<<face.x<<' '<<face.y<<' '<<face.width<<' '<<face.height<<std::endl;
            // }
            // if (orbiting) {
            //     int numVerts = mesh.getNumVertices();
            //     for (int i=0; i<numVerts; ++i) {
            //         ofVec3f vert = mesh.getVertex(i);
            //         float distance = distances[i];
            //         float angle = angles[i];
            //         float elapsedTime = ofGetElapsedTimef() - startOrbitTime;
            //         float speed = ofMap(distance, 0, 200, 1, 0.25, true);
            //         float rotatedAngle = elapsedTime * speed + angle;
            //         vert.x = distance * cos(rotatedAngle) + meshCentroid.x;
            //         vert.y = distance * sin(rotatedAngle) + meshCentroid.y;
            //         mesh.setVertex(i, vert);
            //     }
            // }
            // if (mouseDisplacement) {
            //     ofVec3f mouse(mouseX, ofGetHeight()-mouseY, 0);
            //     for (int i=0; i<mesh.getNumVertices(); ++i) {
            //         ofVec3f vertex = meshCopy.getVertex(i);
            //         float distanceToMouse = mouse.distance(vertex);
            //         float displacement = ofMap(distanceToMouse, 0, 400, 300.0, 0, true);
            //         ofVec3f direction = vertex - mouse;
            //         direction.normalize();
            //         ofVec3f displacedVertex = vertex + displacement*direction;
            //         mesh.setVertex(i, displacedVertex);
            //     }
            // }
            cvtColor(camMat, camMat, COLOR_RGB2BGR);
            vector<BoxInfo> generate_boxes;
            net->detect(camMat, generate_boxes);
            size_t index=0;
            if(generate_boxes.size()>0){
                if(!isInit){
                    isInit = true;
                }else{
                    //sort by confidence
                    for (size_t n = 1; n < generate_boxes.size(); n++){
                        if(generate_boxes[n].score > generate_boxes[index].score){
                            index = n;
                        }
                    }
                    float ratio = distance(generate_boxes[index].kpt3.pt, lastBoxes.kpt3.pt);

                    if(ratio > ratio_thresh){
                        std::string direc = detectMovement(generate_boxes[index], lastBoxes);
                        
                        if(direc == "Right"){
                            cout<<"Right"<<endl;
                            orbiting = 1;
                        }
                        else if(direc == "Left"){
                            cout<<"Left"<<endl;
                            orbiting = 2;
                        }
                        else if(direc == "Up"){
                            cout<<"Up"<<endl;
                            orbiting = 1;
                        }
                        else if(direc == "Down"){
                            cout<<"Down"<<endl;
                            orbiting = 2;
                        }
                        else if(direc == "Closer"){
                            cout<<"Closer"<<endl;
                            orbiting = 1;
                        }
                        else if(direc == "Further"){
                            cout<<"Further"<<endl;
                            orbiting = 2;
                        }
                        // sleep(1);
                        // printf("Rotate\n");
                        // orbiting = 1;
                    }else{
                        orbiting = 0;
                    }
                }
                lastBoxes = generate_boxes[index];
            }else{
                if(isInit){
                    orbiting = 0;
                    isInit = false;
                }else{
                    orbiting = 0;
                }
            }
            if(show_detection_windows){
                imshow(kWinName, camMat);
            }
            
        }
    }
    rotateMesh(mesh, meshCentroid, distances, angles, scaleFactor, startOrbitTime, rotateDirection, clockwise);
    if (orbiting) {

        if(orbiting==1){
            scaleFactor *= 1.25f;
        }else if(orbiting==2){
            scaleFactor *= 0.75f;
        }
        printf("orbiting %d Sacle Factor: %f\n", orbiting, scaleFactor);
    }
    if (mouseDisplacement) {
        // Get the mouse location - it must be relative to the center of our screen
        // because of the ofTranslate() command in draw()
        ofVec3f mouse(mouseX, ofGetHeight()-mouseY, 0);
        
        // Loop through all the vertices in the mesh and move them away from the
        // mouse
        for (int i=0; i<mesh.getNumVertices(); ++i) {
            ofVec3f vertex = meshCopy.getVertex(i);
            float distanceToMouse = mouse.distance(vertex);
            
            // Scale the displacement based on the distance to the mouse
            // A small distance to mouse should yield a small displacement
            float displacement = ofMap(distanceToMouse, 0, 400, 300.0, 0, true);
            
            // Calculate the direction from the mouse to the current vertex
            ofVec3f direction = vertex - mouse;
            
            // Normalize the direction so that it has a length of one
            // This lets us easily change the length of the vector later
            direction.normalize();
            
            // Push the vertex in the direction away from the mouse and push it
            // a distance equal to the value of the variable displacement
            ofVec3f displacedVertex = vertex + displacement*direction;
            mesh.setVertex(i, displacedVertex);
        }
    }
}
//--------------------------------------------------------------
void ofApp::draw(){
    ofColor centerColor = ofColor(85, 78, 68);
    ofColor edgeColor(0, 0, 0);
    ofBackgroundGradient(centerColor, edgeColor, OF_GRADIENT_CIRCULAR);
    easyCam.begin();
    ofPushMatrix();
    ofTranslate(-ofGetWidth()/2 + 240, -ofGetHeight()/2 + 120);
    mesh.draw();
    if(using_video && image_show){
        player.draw(0, 0);
    }else if(!using_video && image_show){
        cam.draw(0, 0);
    }
    if(image_show){
        for(cv::Rect & face : faces) {
            ofNoFill();

            ofDrawRectangle(toOf(face)); // draw face rectangles
        }
    }
    if(show_detection_windows){
        namedWindow(kWinName, WINDOW_NORMAL);
    }else{
        destroyWindow(kWinName);
    }
  
    // model.drawFaces();
    ofPopMatrix();
    easyCam.end();
    
}
//--------------------------------------------------------------
void ofApp::exit(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'a') {
            orbiting = !orbiting; // This inverts the boolean
            startOrbitTime = ofGetElapsedTimef();
            mesh = meshCopy; // This restores the mesh to its original values
        }
    if (key == 'm') {
        mouseDisplacement = !mouseDisplacement; // Inverts the boolean
        mesh = meshCopy; // Restore the original mesh
    }
    
    if(key == 's'){
        show_detection_windows = !show_detection_windows;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
