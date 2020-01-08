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

	string filename = "../wine.csv";
    printf("\nReading in %s...\n\n",filename);
    const double train_test_split_ratio = 0.9;

    Ptr<TrainData> data = TrainData::loadFromCSV(filename, 1, -1, -1, String());

    if( data.empty() )
    {
        printf("ERROR: File %s can not be read\n", filename);
        return 0;
    }

    data->setTrainTestSplitRatio(train_test_split_ratio);

    printf("======BOOST=====\n");
    Ptr<Boost> boost = Boost::create();
    boost->setBoostType(Boost::DISCRETE);//Discrete AdaBoost.
    //boost->setBoostType(Boost::REAL);//Real AdaBoost. It is a technique that utilizes confidence-rated predictions and works well with categorical data.
    //boost->setBoostType(Boost::LOGIT );//LogitBoost. It can produce good regression fits.
    //boost->setBoostType(Boost::GENTLE);//Gentle AdaBoost. It puts less weight on outlier data points and for that reason is often good with regression data
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(10);
    boost->setUseSurrogates(false);
    boost->setPriors(Mat());
    train_and_print_errs(boost, data);
    boost->save("../boost.xml");
 
   

    std::cout << "======TEST====="<<std::endl;
	cv::Ptr<cv::ml::Boost> boost2 =cv::ml::Boost::load("../boost.xml");
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





    float resultKind = boost2->predict(testVec);
    std::cout << "label 1,pred "<<resultKind<<std::endl;
    return 0;
}
