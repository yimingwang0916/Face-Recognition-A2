#include "hog.h"

using namespace std;

cv::Mat histogramEqualizationColor(const cv::Mat& img);

cv::Mat histogramEqualizationColor(const cv::Mat& img) {
  cv::Mat ycrcb;

  cv::cvtColor(img,ycrcb,CV_BGR2YCrCb);

  vector<cv::Mat> channels;
  cv::split(ycrcb,channels);

  cv::equalizeHist(channels[0], channels[0]);

  cv::Mat result;
  cv::merge(channels,ycrcb);

  cv::cvtColor(ycrcb,result,CV_YCrCb2BGR);

  return result;
}

class HOG::HOGPimpl {
public:

  cv::Mat1f descriptors;
  cv::Mat1i responses;
	
  cv::Ptr<cv::ml::SVM> svm;
  cv::Ptr<cv::HOGDescriptor> hog;
};


/// Constructor
HOG::HOG()
{
  pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
}

/// Destructor
HOG::~HOG() 
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
  pimpl->hog = new cv::HOGDescriptor::HOGDescriptor(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1,
    /*winSigma*/ -1, cv::HOGDescriptor::L2Hys, 0.2, 3e-2);
  pimpl->svm = cv::ml::SVM::create();

  pimpl->svm->setType(cv::ml::SVM::C_SVC);
  pimpl->svm->setC(0.3);
  pimpl->svm->setKernel(cv::ml::SVM::INTER);
}

/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const cv::Mat3b& img, bool isPerson)
{
  //    cv::Mat3b imgHE = histogramEqualizationColor(img);
    cv::Mat3b imgHE(img);
	cv::Mat3b img2 = imgHE(cv::Rect((imgHE.cols-64)/2,(imgHE.rows-128)/2,64,128));
	vector<float> vDescriptor;
	pimpl->hog->compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
    
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1i(1, 1, int(isPerson)));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
  // cv::SVMParams params;
  pimpl->svm->train( pimpl->descriptors, cv::ml::ROW_SAMPLE, pimpl->responses );
  //	pimpl->svm.train( pimpl->descriptors, pimpl->responses, cv::Mat(), cv::Mat(), params );
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const cv::Mat3b& img)
{
  //    cv::Mat imgHE = histogramEqualizationColor(img);
      cv::Mat3b imgHE(img);
	cv::Mat3b img2 = imgHE(cv::Rect((imgHE.cols-64)/2,(imgHE.rows-128)/2,64,128));

	vector<float> vDescriptor;
	pimpl->hog->compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	return pimpl->svm->predict(descriptor);
}

