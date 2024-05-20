#include <onnxruntime_cxx_api.h>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include "ofxOpenCv.h"
#include <vector>
#include <iostream>

#include "def.h"

using namespace std;
//using namespace cv;
using namespace Ort;



class HandDetection
{
public:
	HandDetection(Net_config config);
	string detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo_Hand>& input_boxes);
	bool has_postprocess;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "HandDetection");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

HandDetection::HandDetection(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
        AllocatedStringPtr input_name_Ptr  = ort_session->GetInputNameAllocated(i, allocator);
		input_names.push_back(input_name_Ptr.release());
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
        AllocatedStringPtr output_name_Ptr  = ort_session->GetOutputNameAllocated(i, allocator);
		output_names.push_back(output_name_Ptr.release());
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

}

void HandDetection::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

void HandDetection::nms(vector<BoxInfo_Hand>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo_Hand a, BoxInfo_Hand b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo_Hand& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

string HandDetection::detect(Mat& frame)
{
	Mat dstimg;
	resize(frame, dstimg, cv::Size(this->inpWidth, this->inpHeight));
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// ��ʼ����
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	vector<BoxInfo_Hand> generate_boxes;

	Value &predictions = ort_outputs.at(0);
	// Value &labels = ort_outputs.at(1);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	// auto labels_dim = labels.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);

	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score, x1,y1,score1, ...., x5,y5,score5
	const float* pdata = predictions.GetTensorMutableData<float>();
	// const float* ldata = labels.GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///����ͼ�߶�
	{
		float box_score = pdata[4];
		if (box_score > this->nmsThreshold)
		{
			float class_socre = box_score * pdata[5];
			if (class_socre > this->confThreshold)
			{
				float cx = pdata[0] * ratiow;  ///cx
				float cy = pdata[1] * ratioh;   ///cy
				float w = pdata[2] * ratiow;   ///w
				float h = pdata[3] * ratioh;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;
				int label_id = 0;
				if(class_socre>0.5) {
					label_id =1;
				}
				generate_boxes.push_back(BoxInfo_Hand{ xmin, ymin, xmax, ymax, box_score, label_id});
			}
		}
		pdata += nout;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
    int lcounts =0, rcount =0;
	for (size_t n = 0; n < generate_boxes.size(); n++)
	{
		int xmin = int(generate_boxes[n].x1);
		int ymin = int(generate_boxes[n].y1);
		int label_id = generate_boxes[n].label_id;
		rectangle(frame, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[n].x2), int(generate_boxes[n].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[n].score);
		if(label_id==0) {
            rcount++;
			label = "rhand:" + label;
		}else {
            lcounts++;
			label = "lhand:" + label;
		}
		putText(frame, label, cv::Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		// circle(frame, generate_boxes[n].kpt1.pt, 1, Scalar(0, 255, 0), -1);
		// circle(frame, generate_boxes[n].kpt2.pt, 1, Scalar(0, 255, 0), -1);
		// circle(frame, generate_boxes[n].kpt3.pt, 1, Scalar(0, 255, 0), -1);
		// circle(frame, generate_boxes[n].kpt4.pt, 1, Scalar(0, 255, 0), -1);
		// circle(frame, generate_boxes[n].kpt5.pt, 1, Scalar(0, 255, 0), -1);
	}
    if(rcount>0 && lcounts>0){
        return "both";
    }
    else if(rcount>0) {
        return "right";
    }
    else if(lcounts>0) {
        return "left";
    }else {
        return "none";
    }
	// return generate_boxes;
}




HandDetection* get_hand_network(Net_config config) {
	// string modelPath = "/home/cscw/Desktop/YOLOV3ONNX/onnx_weight/yolo_model_hands.onnx";
	config.confThreshold = -1;
	HandDetection *mynet = new HandDetection(config);
    return mynet;
// 	std::string imgpath = "/home/cscw/Desktop/YOLOV3ONNX/images/alexis-brown-omeaHbEFlN4-unsplash.jpg";
// 	Mat srcimg = imread(imgpath);
// 	std::vector<BoxInfo> faceboxes = mynet.detect(srcimg);
// 	std::string winname = "Deep learning Head Pose Estimation in ONNXRuntime";
// 	namedWindow(winname, WINDOW_AUTOSIZE);
// 	resizeWindow(winname, 640, 480);
// 	imshow(winname, srcimg);
// 	waitKey();
// 	// drawPred(srcimg, faceboxes);

// 	// std::string winname = "Deep learning Head Pose Estimation in ONNXRuntime";
// 	VideoCapture cap = VideoCapture(0);

// 	while(1) {
// 		Mat srcimg;
// 		cap>>srcimg;
// 		if(srcimg.empty()) {
// 			break;
// 		}
// 		// resize(srcimg, srcimg, Size(640, 480));
// 		resize(srcimg, srcimg, Size(640, 480));
// 		std::vector<BoxInfo> faceboxes = mynet.detect(srcimg);
// 		// drawPred(srcimg, faceboxes);

// 		namedWindow(winname, WINDOW_AUTOSIZE);
// 		resizeWindow(winname, 640, 480);
// 		imshow(winname, srcimg);
// 		waitKey(1);
// 		// waitKey();
// 	}
// 	destroyAllWindows();
// 	return 0;
}
