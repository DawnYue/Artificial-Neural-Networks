#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include <dnn.hpp>

//using namespace dnn;
using namespace cv;
using namespace std;
//课前准备
//通过非极大值抑制去掉置信度较低的bouding box
void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);

// 获得输出名字
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);


//绘制检测结果
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

void yoloV3();


std::vector<std::string> classes;

float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<cv::String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}


	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));

	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	else
	{
		std::cout << "classes is empty..." << std::endl;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
}

//opencv 调用 yolov3 demo
void yoloV3()
{

	//coco数据集的名称文件，80类
	string classesFile = "H:\\coco.names";
	//yolov3网络模型文件
	String yolov3_model = "H:\\yolov3.cfg";
	//权重
	String weights = "H:\\yolov3.weights";

	int in_w, in_h;
	double thresh = 0.5;
	double nms_thresh = 0.25;
	in_w = in_h = 608;

	//将coco.names中的80类名称转换为vector形式
	std::ifstream classNamesFile(classesFile.c_str());
	if (classNamesFile.is_open())
	{
		std::string className = "";
		// getline (istream&  is, string& str)
		//is为输入，从is中读取读取的字符串保存在string类型的str中，如果没有读入字符返回false，循环结束
		while (std::getline(classNamesFile, className)) {
			classes.push_back(className);
		}
	}
	else {
		std::cout << "can not open classNamesFile" << std::endl;
	}

	cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolov3_model, weights);

//	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
//	net.setPreferableTarget(DNN_TARGET_CPU);








}
int main()
{
	//开始计时
	double start = static_cast<double>(cvGetTickCount());


		//yoloV3();
	Mat img = imread("E://1//1.png");
	imshow("test", img);
	//等待用户按键
	waitKey(0);
	

	//结束计时
	double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
	//显示时间
	cout << "processing time:" << time / 1000 << "ms" << endl;

	//等待键盘响应，按任意键结束程序
	system("pause");
	return 0;
}