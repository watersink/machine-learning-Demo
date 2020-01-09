#include <iostream>
#include <fstream>
#include <sstream>
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
using namespace std;
using namespace cv;
using namespace cv::ml;


int read_csv(string csv_filename,int classnum,  float datas[][13],float labels[][3]){


	ifstream inFile(csv_filename,ios::in);//inFile来自fstream,ifstream为输入文件流(从文件读入)
	string lineStr;
	
	int linenum=0;
	while(getline(inFile,lineStr)) //getline来自sstream
	{
		if (linenum==0){
			linenum++;
			continue;
		}
		stringstream ss(lineStr);//来自sstream
		string str;
		int strnum=0;
		while(getline(ss,str,',')){
			if (strnum==13){
				for(int i=0;i<classnum;i++){
					if (std::stoi(str)==i)
						labels[linenum][i]=1.0;
					else
						labels[linenum][i]=0.0;
				}
			}else{
				datas[linenum][strnum]=atof(str.c_str());//一行数据以vector保存
			}

			strnum+=1;
		}



		linenum++;

	}


	return 0;

}



int main(int argc, char** argv){
	//不需要归一化，不需要shuffle,label必须得是onehot,类别必须从0开始，
	string filename ="../wine.csv";

	int classnum=3;//3分类
	int linenum=178;
	float datas[178][13]={ { 0 } };
	float labels[178][3]={ { 0 } };
	read_csv(filename,classnum, datas,labels);


	/*
	for (int i =0;i<linenum;i++){
		for (int j=0;j<13;j++)
			cout<<datas[i][j]<<" ";
		cout<<endl;
	}*/

	int input_size = 13;
    const double train_test_split_ratio = 0.7;
	Mat train_data_mat(linenum, input_size, CV_32FC1, &datas);
	Mat labels_mat(linenum, classnum, CV_32FC1, &labels);
	///cout<<labels_mat<<endl;
	Ptr<TrainData> trainData = TrainData::create(train_data_mat, ml::ROW_SAMPLE, labels_mat);
    trainData->setTrainTestSplitRatio(train_test_split_ratio);


	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1,5)<<input_size,16,64,128,classnum);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000,0.0001));
	model->train(trainData); 

    printf( "train error: %f\n", model->calcError(trainData, false, noArray()) );
    printf( "test error: %f\n\n", model->calcError(trainData, true, noArray()) );


	//保存训练结果
	model->save("../mlp.xml");


    //test
	Ptr<ANN_MLP>model_t = ANN_MLP::create();
	model_t = cv::Algorithm::load<cv::ml::ANN_MLP>("../mlp.xml");
	float testdata[13] = {14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065};//0
	//float testdata[13] = {11.65,1.67,2.62,26,88,1.92,1.61,.4,1.34,2.6,1.36,3.21,562};//1
    //float testdata[13] = {12.93,2.81,2.7,21,96,1.54,.5,.53,.75,4.6,.77,2.31,600};//2

	cv::Mat testMat(1, input_size, CV_32FC1, testdata);
	cv::Mat dst;
	float resultKind= model_t->predict(testMat, dst);


	cout << "dst:" << dst << endl;
	//选出最大值
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	cout << "测试结果：" << maxLoc.x << "置信度:" << maxVal * 100 << "%" << endl;





	return 0;
}
