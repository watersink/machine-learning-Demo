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

	string filename = "../air.csv";
    printf("\nReading in %s...\n\n",filename);
    const double train_test_split_ratio = 0.7;

    Ptr<TrainData> data = TrainData::loadFromCSV(filename, 0, -1, -1, String());

    if( data.empty() )
    {
        printf("ERROR: File %s can not be read\n", filename);
        return 0;
    }

    data->setTrainTestSplitRatio(train_test_split_ratio);

    printf("======DTREE=====\n");
	int depth =13;
	for (int i=0;i<depth;i++){
	    Ptr<DTrees> dtree = DTrees::create();
    	dtree->setMaxDepth(i);
    	dtree->setMinSampleCount(2);
    	dtree->setRegressionAccuracy(0);
    	dtree->setUseSurrogates(false);
    	dtree->setMaxCategories(3);
    	dtree->setCVFolds(1);
    	dtree->setUse1SERule(true);
    	dtree->setTruncatePrunedTree(true);
    	dtree->setPriors(Mat());

		cout<<"depth "<<i<<" "<<endl;
    	train_and_print_errs(dtree, data);
    	dtree->save("dtree_result.xml");
	}


   

    std::cout << "======TEST====="<<std::endl;
	cv::Ptr<cv::ml::DTrees> dtree2 =cv::ml::DTrees::load("dtree_result.xml");
    std::vector<float>testVec;
    testVec.push_back(0.10111413);
    testVec.push_back(0.147943);
    testVec.push_back(0.1385576);
    testVec.push_back(0.35972223);
    float resultKind = dtree2->predict(testVec);
    std::cout << "label 2,pred "<<resultKind<<std::endl;
    return 0;
}
