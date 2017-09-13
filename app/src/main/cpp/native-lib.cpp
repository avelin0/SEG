#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

extern "C"{

#define MB_THRESHOLD 20
using namespace std;
using namespace cv;

JNIEXPORT jstring JNICALL
Java_com_example_bruno_seg_CameraManip_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

void JNICALL Java_com_example_bruno_seg_CameraManip_salt(JNIEnv *env, jobject instance,
                                                          jlong matAddrGray,
                                                          jint nbrElem){
    cv::Mat &mGr=*(cv::Mat*) matAddrGray;
    for (int k = 0; k < nbrElem; ++k) {
        int i=rand()%mGr.cols;
        int j=rand()%mGr.rows;
        mGr.at<uchar>(j,i)=255;
    }
}

JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_morphoOp(
        JNIEnv *env, jobject obj,
        jlong addrSrc, jlong addrDst,
        int morph_elem = 0,int morph_size = 1,int morph_operator = 1, std::string mensagem="Morpho OP"){

    cv::Mat& mSrc  = *(cv::Mat*)addrSrc;
    cv::Mat& mDst = *(cv::Mat*)addrDst;
    int operation = morph_operator + 2;// Since MORPH_X : 2,3,4,5 and 6
    cv::Mat element = cv::getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( mSrc, mDst, operation, element );
}

JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_FindFeatures(JNIEnv*, jobject,
                                                                            jlong addrGray,
                                                                            jlong addrRgba) {
    cv::Mat& mGr  = *(cv::Mat*)addrGray;
    cv::Mat& mRgb = *(cv::Mat*)addrRgba;
    std::vector<cv::KeyPoint> v;

    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(50);
    detector->detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ ) {
        const cv::KeyPoint& kp = v[i];
        cv::circle(mRgb, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }
}
JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_thresh(JNIEnv *env, jobject obj,
                                                                        jlong scrGray,
                                                                        jlong dst,
                                                                     double thresholdValue=2.0,
                                                                     double maxVal=255.0){
    cv::Mat matDst = *((cv::Mat*)dst);
    cv::Mat matSrcGray = *((cv::Mat*)scrGray);
    threshold( matSrcGray, matDst, thresholdValue, maxVal, CV_THRESH_BINARY);
}

JNIEXPORT cv::Mat JNICALL Java_com_example_bruno_seg_GalleryActivity_sobel(
        JNIEnv *env, jobject obj,
        cv::Mat src_gray,int scale = 1,int delta = 0,int ddepth = CV_16S,
        std::string mensagem="Sobel Image"){

    cv::Mat grad;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 1., abs_grad_y, 1., 0, grad );
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            if (grad.at<uchar>(i,j) < MB_THRESHOLD)
                grad.at<uchar>(i,j)=0;
        }
    }

//    cv::show_image(grad,mensagem);
    return grad;
}



JNIEXPORT cv::Mat JNICALL Java_com_example_bruno_seg_GalleryActivity_to_gray(
        JNIEnv *env, jobject obj,
        jlong src,std::string mensagem="To Gray"){
    cv::Mat matDst = *((cv::Mat*)src);
    cv::cvtColor(matDst, matDst, CV_BGR2GRAY);
//    show_image(src,mensagem);
    return matDst;
}

}


