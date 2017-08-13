#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

extern "C"{
JNIEXPORT jstring JNICALL
Java_com_example_bruno_seg_CameraManip_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

jstring
Java_com_example_bruno_seg_CameraManip_validate(JNIEnv *env,jobject,jlong addrGray,jlong addrRgba){
    cv::Rect();
    cv::Mat();
    std::string hello2="Hello from validate";
    return env->NewStringUTF(hello2.c_str());
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

JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_FindFeatures(JNIEnv*, jobject,
                                                                            jlong addrGray,
                                                                            jlong addrRgba) {
    cv::Mat& mGr  = *(cv::Mat*)addrGray;
    cv::Mat& mRgb = *(cv::Mat*)addrRgba;
    std::vector<cv::KeyPoint> v;

    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(50);
    detector->detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const cv::KeyPoint& kp = v[i];
        circle(mRgb, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }
}
}


