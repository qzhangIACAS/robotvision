
#include"InRobot.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
using namespace InRobot;
using namespace std;
using namespace cv;

#define MinSize 80   //定义检测程序中检测窗口的最小尺寸，单位为像素
#define MaxSize 800   //定义检测程序中检测窗口的最大尺寸，单位为像素

InRobotVision::InRobotVision()
{

}
void InRobotVision::NestDetect(Mat frame)
{
	
	Mat Image = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	Mat seg;
	//如果读入图像失败
	if (frame.empty())
	{
		fprintf(stderr, "Can not load image \n");

		exit(-1);
	}
	frame_img = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	resize(frame, frame,Size(640,480));
	cvtColor(frame, Image, CV_BGR2GRAY);
	threshold(Image, frame_img, 30, 255, CV_THRESH_BINARY_INV);

	//形态学闭操作 填补前景中的细小空洞  
	dilate(frame_img, frame_img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(frame_img, frame_img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//查找轮廓
	findContours(frame_img, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//	Mat result(frame.size(), CV_8U, Scalar(255));
	//画出轮廓
	//drawContours(result, contours, -1, Scalar(0), 2);


	// iterate through all the top-level contours,
	// draw each connected component with its own random color

	std::vector<cv::Point> approx;

	frame_dst = frame.clone();

	for (size_t i = 0; i < contours.size(); i++)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(
			cv::Mat(contours[i]),
			approx,
			cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
			true
			);

		// 如果轮廓面积大于300并且轮廓是非凸的，则判断当前轮廓为鸟巢
		if (std::fabs(cv::contourArea(contours[i])) > 300 && !cv::isContourConvex(approx))
			setLabel(frame_dst, "Bird Nest", contours[i]);

	}

	
	

}
void InRobotVision::InsulatorDetect(Mat frame)
{

	Mat Image;// = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	//从文件中读入图像

	frame_img;// = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	//如果读入图像失败
	if (frame.empty())
	{
		fprintf(stderr, "Can not load image %s\n");
		exit(-1);
	}
	//resize(frame,frame,Size(640,480));
	//GaussianBlur(frame, frame, Size(5, 5), 0, 0);
	cvtColor(frame, Image, CV_BGR2GRAY);
	equalizeHist(Image, Image);
	threshold(Image, frame_img, 80, 255, CV_THRESH_BINARY_INV);
	
	//morphological opening (remove small objects from the foreground)
	//erode(seg, seg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	//dilate(seg, seg, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

	//morphological closing (fill small holes in the foreground)
	dilate(frame_img, frame_img, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
	erode(frame_img, frame_img, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
	
	//Laplacian(seg, seg, -1, 5);
	vector<vector<Point>> contours; //存储提取的轮廓
	vector<Vec4i> hierarchy;
	findContours(frame_img, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	if (contours.size() == 0)
		exit(-8);
	std::vector<cv::Point> approx;

	// We'll put the labels in this destination image
	frame_dst = frame.clone();
	/// Find the convex hull object for each contour  
	vector<vector<Point> >hull(contours.size());
	// Int type hull  
	vector<vector<int>> hullsI(contours.size());
	// Convexity defects  
	vector<vector<Vec4i>> defects(contours.size());


	for (size_t i = 0; i < contours.size(); i++)
	{

		convexHull(Mat(contours[i]), hull[i], false);
		// find int type hull  
		convexHull(Mat(contours[i]), hullsI[i], false);
		// get convexity defects  
		if (hullsI[i].size() <= 10)
			continue;
		convexityDefects(Mat(contours[i]), hullsI[i], defects[i]);

	}
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		// draw defects
		//size_t count = contours[i].size();
		//std::cout << "Count : " << count << std::endl;

		vector<Vec4i>::iterator d = defects[i].begin();

		while (d != defects[i].end()) {
			Vec4i& v = (*d);
			// if(IndexOfBiggestContour == i)
			{

				int startidx = v[0];
				Point ptStart(contours[i][startidx]); // point of the contour where the defect begins
				int endidx = v[1];
				Point ptEnd(contours[i][endidx]); // point of the contour where the defect ends
				int faridx = v[2];
				Point ptFar(contours[i][faridx]);// the farthest from the convex hull point within the defect
				int depth = v[3] / 256; // distance between the farthest point and the convex hull

				if (depth >23 && depth < 30)
				{
					line(frame_dst, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
					line(frame_dst, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
					circle(frame_dst, ptStart, 4, Scalar(255, 0, 100), 2);
					circle(frame_dst, ptEnd, 4, Scalar(255, 0, 100), 2);
					circle(frame_dst, ptFar, 4, Scalar(100, 0, 255), 2);
					putText(frame_dst, "Damage!!!", Point(contours[i][startidx]), CV_FONT_HERSHEY_COMPLEX, frame_dst.cols / 400, cvScalar(200, 200, 200, 0));
				}


			}
			d++;
		}


	}

	
}
void InRobotVision::PowerTransDetect(Mat frame,string cascadeFilePath)
{
	
	
	//-- 1. 下载分类器
	if (!powertransformer_cascade.load(cascadeFilePath))
	{
		std::cout << "--分类器下载出错！！\n";
		exit(-1);
	}
	

		if (frame.empty())
		{
			std::cout << "--图片下载出错！！\n";
			exit(-2);
		}
		
		std::vector<Rect> trans;//定义存储检测结果的矩形向量
		Mat frame_gray;//
		
		GaussianBlur(frame, frame, Size(1, 1), 0, 0);

		resize(frame, frame, Size(640, 480), INTER_AREA);
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//将图片转换为灰度图像

		equalizeHist(frame_gray, frame_gray);//直方图均衡化，增强图像的亮度及对比度
		frame_dst = frame.clone();
		//-- 检测变压器
		powertransformer_cascade.detectMultiScale(frame_gray, trans, 1.1, 2, 0, Size(MinSize, MinSize), Size(MaxSize, MaxSize));//检测结果存储到trans向量中，忽略（80，80）--（300，300）以外的窗口

		for (size_t i = 0; i < trans.size(); i++)//
		{
			if (trans.size() == 0)
				break;

			Point center(trans[i].x + trans[i].width / 2, trans[i].y + trans[i].height / 2);//求得矩形窗口的中心点
			ellipse(frame_dst, center, Size(trans[i].width / 2, trans[i].height / 2), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);//在图片上标记椭圆
			rectangle(frame_dst, trans[i], Scalar(255, 0, 255));//在图片上标记矩形


		}

		
			
		
		
	
	
}
int InRobotVision::FlagDetect(Mat frame)
{
	return 0;
}
void InRobotVision::WireDetect(Mat frame)
{
	
	Mat Image = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	//从文件中读入图像
	
	Mat src, seg, blur_img, hist_img, threshold_img, canny_img, dilate_img, erode_img;

	//如果读入图像失败
	if (frame.empty())
	{
		cout<<"Can not load image %s\n"<<endl;
		exit(-1);
	}
	resize(frame, src, Size(640, 480));
	
	GaussianBlur(src, blur_img, Size(5, 5), 0, 0);
	cvtColor(blur_img, Image, CV_BGR2GRAY);
	equalizeHist(Image, hist_img);
	threshold(hist_img, threshold_img, 40, 255, CV_THRESH_BINARY_INV);
	
	Canny(threshold_img, canny_img, 40, 120);

	//morphological closing (fill small holes in the foreground)
	dilate(threshold_img, dilate_img, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(dilate_img, erode_img, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(erode_img, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	
	if (contours.size() == 0)
		exit(-8);
	
	// We'll put the labels in this destination image
	frame_dst = src.clone();
	/// Find the convex hull object for each contour  
	// Int type hull  
	vector<vector<int>> hullsI(contours.size());
	// Convexity defects  
	vector<vector<Vec4i>> defects(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		// find int type hull  
		convexHull(Mat(contours[i]), hullsI[i], false);
		// get convexity defects  
		if (hullsI[i].size() <= 10)
			continue;
		convexityDefects(Mat(contours[i]), hullsI[i], defects[i]);

	}
	for (size_t i = 0; i < contours.size(); i++)
	{
		// draw defects
		vector<Vec4i>::iterator d = defects[i].begin();
		while (d != defects[i].end()) {
			Vec4i& v = (*d);
			// if(IndexOfBiggestContour == i)
			{

				int startidx = v[0];
				Point ptStart(contours[i][startidx]); // point of the contour where the defect begins
				int endidx = v[1];
				Point ptEnd(contours[i][endidx]); // point of the contour where the defect ends
				int faridx = v[2];
				Point ptFar(contours[i][faridx]);// the farthest from the convex hull point within the defect
				int depth = v[3] / 256; // distance between the farthest point and the convex hull

				if (depth >10 && depth < 60)
				{
					line(frame_dst, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
					line(frame_dst, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
					circle(frame_dst, ptStart, 4, Scalar(255, 0, 100), 2);
					circle(frame_dst, ptEnd, 4, Scalar(255, 0, 100), 2);
					circle(frame_dst, ptFar, 4, Scalar(100, 0, 255), 2);
					putText(frame_dst, "Damage!!!", Point(contours[i][startidx]), CV_FONT_HERSHEY_COMPLEX, frame_dst.cols / 400, cvScalar(0, 0, 255, 0));
				}

			}
			d++;
		}

	}

	
}
void InRobotVision::MeterDetect(Mat frame)
{
	Mat src, result, gray_img,blur_img, hist_img, threshold_img, canny_img, dilate_img, erode_img, flip_img;

	//如果读入图像失败
	if (frame.empty())
	{
		cout<<"Can not load image %s\n"<<endl;
		exit(-1);
	}
	resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));

	transpose(frame, src);
	flip(src, flip_img, 0);


	bilateralFilter(flip_img, blur_img, 5, 5 * 2, 5 / 2);

	cvtColor(flip_img, gray_img, CV_BGR2GRAY);

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(gray_img, circles, CV_HOUGH_GRADIENT, 1, gray_img.rows / 1, 300, 50, 0, 0);


	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		
		// circle outline
		circle(flip_img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		//
		Point pt((int)circles[i][0] - (int)circles[i][2], (int)circles[i][1] - (int)circles[i][2]);
		Mat ROI(gray_img, Rect(((int)circles[i][0] - (int)circles[i][2]), ((int)circles[i][1] - (int)circles[i][2]), radius * 2, radius * 2));
		resize(ROI, ROI, Size(ROI.cols * 4, ROI.rows * 4));
		frame_dst = ROI.clone();

		equalizeHist(ROI, hist_img);

	    threshold(hist_img, threshold_img, 165, 255, CV_THRESH_BINARY_INV);
		
		Canny(threshold_img, canny_img, 50, 150, 3);
		dilate(canny_img, dilate_img, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
		erode(dilate_img, erode_img, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
		
		cvtColor(frame_dst, frame_dst, CV_GRAY2BGR);
		// We'll put the labels in this destination image
		
		std::vector<cv::Vec4i> lines;
		//霍夫变换,获得一组极坐标参数（rho，theta）,每一对对应一条直线，保存到lines    
		//第3,4个参数表示在（rho，theta)坐标系里横纵坐标的最小单位，即步长    
		cv::HoughLinesP(erode_img, lines, 1, CV_PI / 180, 35, 17, 0);
		for (size_t i = 0; i < lines.size(); i++)
		{
			Point pt1(lines[i][0], lines[i][1]);
			Point pt2(lines[i][2], lines[i][3]);
			line(frame_dst, pt1, pt2, cv::Scalar(0, 0, 255), 2);
			float k = (float)(lines[i][3] - lines[i][1]) / (float)(lines[i][2] - lines[i][0]);
			float theta = atanf(-k);
			cout << "起始点： X " << lines[i][0] << " Y " << lines[i][1] << endl;
			cout << "终止点： X " << lines[i][2] << " Y " << lines[i][3] << endl;
			//cout << "Meter Theta: " << (theta/CV_PI)*180 << endl;
			if (abs(k)>1)
				cout << "Meter Abanomal!!! The Meter Theta:" << (theta / CV_PI) * 180 << endl;
			else
				cout << "Meter Normal!!!  The Meter Theta : " << (theta / CV_PI) * 180 << endl;
		}
	
	}




}
void InRobotVision::readDirectory(const string& directoryName, vector<string>& filenames, bool addDirectoryName)
{
	filenames.clear();

#if defined(WIN32) | defined(_WIN32)
	struct _finddata_t s_file;
	string str = directoryName + "\\*.*";

	intptr_t h_file = _findfirst(str.c_str(), &s_file);
	if (h_file != static_cast<intptr_t>(-1.0))
	{
		do
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "\\" + s_file.name);//将文件名写入字符串向量filenames中
			else
				filenames.push_back((string)s_file.name);//将文件名写入字符串向量filenames中
		} while (_findnext(h_file, &s_file) == 0);
	}
	_findclose(h_file);
#else
	DIR* dir = opendir(directoryName.c_str());
	if (dir != NULL)
	{
		struct dirent* dent;
		while ((dent = readdir(dir)) != NULL)
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "/" + string(dent->d_name));
			else
				filenames.push_back(string(dent->d_name));
		}

		closedir(dir);
	}
#endif

	sort(filenames.begin(), filenames.end());
}

void InRobotVision::setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255, 255, 255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}
InRobotVision::~InRobotVision()
{

}
