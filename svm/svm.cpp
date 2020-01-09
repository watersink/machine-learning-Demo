#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
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


int main(int, char**)
{


	string filename = "../wine.csv";
    printf("\nReading in %s...\n\n",filename);
    const double train_test_split_ratio = 0.7;

    Ptr<TrainData> data = TrainData::loadFromCSV(filename, 1, -1, -1, String());

    if( data.empty() )
    {
        printf("ERROR: File %s can not be read\n", filename);
        return 0;
    }

    data->setTrainTestSplitRatio(train_test_split_ratio);



    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    train_and_print_errs(svm, data);

    svm->save("../svm.xml");
 

 	cv::Mat sv = svm->getUncompressedSupportVectors();
	cout<<"支持向量: ";
    for (int i = 0; i < sv.rows; i++){
		const float* v = sv.ptr<float>(i);
		cout<<v[i]<<"  ";
	}
	cout<<endl;

   

    std::cout << "======TEST====="<<std::endl;
	cv::Ptr<cv::ml::SVM> svm2 =cv::ml::SVM::load("../svm.xml");
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





    float resultKind = svm2->predict(testVec);
    std::cout << "label 1,pred "<<resultKind<<std::endl;

    return 0;
}
