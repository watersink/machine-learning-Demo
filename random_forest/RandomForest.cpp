#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;


static void train_and_print_errs(Ptr<StatModel> model, const Ptr<TrainData>& data)
{
    bool ok = model->train(data);
    if( !ok )
    {
        printf("Training failed\n");
    }
    else
    {
        printf( "train error: %f\n", model->calcError(data, false, noArray()) );
        printf( "test error: %f\n\n", model->calcError(data, true, noArray()) );
    }
}

int main(int argc, char** argv)
{
    string filename="../wine.csv";
    printf("\nReading in %s...\n\n",filename);
    const double train_test_split_ratio = 0.7;

    Ptr<TrainData> data = TrainData::loadFromCSV(filename, 1,-1,-1 , String());

    if( data.empty() )
    {
        printf("ERROR: File %s can not be read\n", filename);
        return 0;
    }

    data->setTrainTestSplitRatio(train_test_split_ratio);

	int depth =13;
    printf("======RTREES=====\n");
    Ptr<RTrees> rtrees = RTrees::create();
    rtrees->setMaxDepth(10);
    rtrees->setMinSampleCount(2);
    rtrees->setRegressionAccuracy(0);
    rtrees->setUseSurrogates(false);
    rtrees->setMaxCategories(3);
    rtrees->setPriors(Mat());
    rtrees->setCalculateVarImportance(false);
    rtrees->setActiveVarCount(0);
    rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0));
		
    train_and_print_errs(rtrees, data);
    rtrees->save("../rtree.xml");


   //随机森林中的树个数
   cout << "Number of trees: " << rtrees->getRoots().size() << endl;
   // 变量重要性
   Mat var_importance = rtrees->getVarImportance();
   if( !var_importance.empty() ){
       double rt_imp_sum = sum( var_importance )[0];
       printf("var#\timportance (in %%):\n");
       int i, n = (int)var_importance.total();
       for( i = 0; i < n; i++ )
           printf( "%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i)/rt_imp_sum);
   }






    std::cout << "======TEST====="<<std::endl;
	cv::Ptr<cv::ml::RTrees> rtree2 =cv::ml::RTrees::load("../rtree.xml");
    std::vector<float>testVec;
    testVec.push_back(14.23);
    testVec.push_back(1.71);
    testVec.push_back(2.43);
    testVec.push_back(15.6);
    testVec.push_back(127);
    testVec.push_back(2.8);
    testVec.push_back(3.06);
    testVec.push_back(0.28);
    testVec.push_back(2.29);
    testVec.push_back(5.64);
    testVec.push_back(1.04);
    testVec.push_back(3.92);
    testVec.push_back(1065);
    float resultKind = rtree2->predict(testVec);
    std::cout << "label 1,pred "<<resultKind<<std::endl;
    return 0;
}
