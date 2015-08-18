#ifndef INROBOT_H_
#define INROBOT_H_
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <stdio.h>
#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif
using namespace cv;
namespace InRobot
{
	
	class InRobotVision
	{
	public:
		InRobotVision();
		~InRobotVision();
		void NestDetect(Mat frame);     //鸟巢检测函数
		void InsulatorDetect(Mat frame);//绝缘子破损检测
		void PowerTransDetect(Mat frame,string cascadeFilePath);
		void WireDetect(Mat frame);     //预绞丝散股检测
		int FlagDetect(Mat frame);     //刀闸开合标志检测
		void MeterDetect(Mat frame);
		void readDirectory(const string& directoryName, vector<string>& filenames, bool addDirectoryName=true);
		void setLabel(Mat& im, const std::string label, std::vector<cv::Point>& contour);
		Mat frame_dst;
	private:
		Mat frame_img;
		CascadeClassifier powertransformer_cascade;
		
	};

}


#endif
