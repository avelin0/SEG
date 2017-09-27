#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <android/bitmap.h>
#include <android/log.h>

extern "C"{

using namespace std;
using namespace cv;

Mat lab;
Mat val;
Mat img;
int show=0;
int save=0;

int MB_THRESHOLD = 20;

//Watersheed
const float INIT = 0.0;
float max_pixel;
float VMAX;
float LMAX;
float new_label = 0.0;
int changed = 0;
int scan_step2 = 1;
int scan_step3 = 1;
int windowSize = 1;

void printInt(string ident,int number){
    stringstream strst;
    strst<<number;
    __android_log_print(ANDROID_LOG_VERBOSE, ident.c_str(), strst.str().c_str(), 1);
}

void printMat(Mat mat, string str){
    stringstream mss;
    mss<<"Linhas: "<<mat.rows<< " Colunas: "<<mat.cols<<" Canais: "<<mat.channels()<<" Type: "<<mat.type();
    __android_log_print(ANDROID_LOG_INFO, str.c_str(), mss.str().c_str(), 1);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            stringstream ss;
            ss<<"("<<i<<","<<j<<") = "<< mat.at<float>(i,j);
            __android_log_print(ANDROID_LOG_INFO, str.c_str(), ss.str().c_str(), 1);
        }

    }
}

/*! \brief color_watershed
 *
 *  Colore baseado no resultado do watershed. Como apos o watershed tem-se uma matriz na escala
 *  de 0 a 255, caso os elementos vizinhos dessa matriz estejam com o mesmo valor, a cor sera a mesma,
 *  caso não, colore com cores diferentes.
 *
 *  Obs: este algoritmo encontra-se na propria documentacao do codigo watershed implementado
 *      no site do opencv
 *
 * \param Mat imagem em formato da matriz a ser manipulada
 * \param int comprimento da janela, valor default = 1
 * \param string Identifica a imagem na janela
 * \return Mat
 */
Mat color_watershed(Mat color) {
    vector<Vec3b> colors;
    for (size_t i = 0; i < 255; i++) {
        int b = theRNG().uniform(10, 255);
        int g = theRNG().uniform(10, 255);
        int r = theRNG().uniform(10, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }
    Mat dst = Mat::zeros(color.size(), CV_8UC3);

    for (int i = 0; i < color.rows; i++) {
        for (int j = 0; j < color.cols; j++) {

            color.convertTo(color, CV_32F);
            int index = color.at<int>(i, j) % (int) 255;
            if (index > 0 && index <= 255) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            } else
                dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }
    return dst;
}

Mat morpho_op(Mat src,int morph_elem = 0,int morph_size = 1,int morph_operator = 1,string mensagem="Morpho OP"){
    Mat dst;
    int operation = morph_operator + 2;
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

    morphologyEx( src, dst, operation, element );
    return dst;
}

Mat sobel(Mat src_gray,int sobelThreshold,int scale = 1,int delta = 0,int ddepth = CV_16S,string mensagem="Sobel Image"){
    Mat grad;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    if( sobelThreshold > 0)
        MB_THRESHOLD=sobelThreshold;

    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 1., abs_grad_y, 1., 0, grad );
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            if (grad.at<uchar>(i,j) < MB_THRESHOLD)
                grad.at<uchar>(i,j)=0;
        }
    }

    return grad;
}

/*! \brief Compare Float
 *
 *  Compara variaveis nao inteiras
 *
 * \param float primeira variavel a ser comparada
 * \param float segunda variavel a ser comparada
 * \param float constante de comparacao(valor default = 0.005f)
 * \return bool
 */
bool cmpf(float a, float b, float epsilon = 0.005f) {
    return (fabs(a - b) < epsilon);
}

/*! \brief Step 1
 *
 *  A menor vizinhança de cada pixel é encontrada. Percorre os vizinhos de um ponto p(x,y), e
 *  se houver algum vizinho menor, o valor do pixel iterado é 1.
 *
 * \param x posicao no eixo x
 * \param y posicao no eixo y
 * \return void
 */
void step1(int x, int y) {
    if (val.at<float>(x, y) != 1) {
        for (int i =- windowSize; i <= windowSize; i++) {
            for (int j =- windowSize; j <= windowSize; j++) {
                if (!(i == 0 && j == 0)) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1) {

                        if (img.at<float>(x, y) > img.at<float>(x + i, y + j)) {
                            val.at<float>(x, y) = 1.0;
                        }
                    }
                }
            }
        }
    }
}

/*! \brief Step 2
 *
 *  Se o pixel está no plateau e seus vizinhos apontam para um dos minimos locais, entao
 *
 * \param x posicao no eixo x
 * \param y posicao no eixo y
 * \return void
 */
void step2(int x, int y) {
    if (val.at<float>(x, y) != 1) {
        float min = VMAX;
        for (int i = -windowSize; i <= windowSize; i++) {
            for (int j = -windowSize; j <= windowSize; j++) {
                if (!(i == 0 && j == 0)) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1) {
                        if (img.at<float>(x, y) == img.at<float>(x + i, y + j) && val.at<float>(x + i, y + j) > 0 &&
                            val.at<float>(x + i, y + j) < min) {
                            min = val.at<float>(x + i, y + j);
                        }
                    }
                }
            }
        }
        if (min != VMAX && val.at<float>(x, y) != min + 1) {
            val.at<float>(x, y) = min + 1.0;
            if (min + 1.0 > LMAX) {
                LMAX = min + 1.0;
            }
            changed = 1;
        }
    }
}

/*! \brief Step 3
 *
 * Labels são atribuídos ao array l[p]
 *
 * \param x posicao no eixo x
 * \param y posicao no eixo y
 * \return void
 */
void step3(int x, int y) {
    float lmin = LMAX;
    float fmin = img.at<float>(x, y);
    if (cmpf(val.at<float>(x, y), 0.0)) {
        for (int i = -windowSize; i <= windowSize; i++) {
            for (int j = -windowSize; j <= windowSize; j++) {
                if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1 && !(i == 0 && j == 0)) {
                    if (cmpf(img.at<float>(x, y), img.at<float>(x + i, y + j)) && lab.at<float>(x + i, y + j) > 0.0 &&
                        lab.at<float>(x + i, y + j) < lmin) {
                        lmin = lab.at<float>(x + i, y + j);
                    }
                }
            }
        }
        if (cmpf(lmin, LMAX) && cmpf(lab.at<float>(x, y), 0.0)) {
            lmin = ++new_label;
        }
    } else {
        if (cmpf(val.at<float>(x, y), 1.0)) {
            for (int i = -windowSize; i <= windowSize; i++) {
                for (int j = -windowSize; j <= windowSize; j++) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1 &&
                        !(i == 0 && j == 0)) {
                        if (img.at<float>(x + i, y + j) < fmin) {
                            fmin = img.at<float>(x + i, y + j);
                        }
                    }
                }
            }

            for (int i = -windowSize; i <= windowSize; i++) {
                for (int j = -windowSize; j <= windowSize; j++) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1 &&
                        !(i == 0 && j == 0)) {
                        if (cmpf(img.at<float>(x + i, y + j), fmin) && lab.at<float>(x + i, y + j) > 0.0 &&
                            lab.at<float>(x + i, y + j) < lmin) {
                            lmin = lab.at<float>(x + i, y + j);
                        }
                    }
                }
            }
        } else {
            for (int i = -windowSize; i <= windowSize; i++) {
                for (int j = -windowSize; j <= windowSize; j++) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows - 1 && y + j <= img.cols - 1 &&
                        !(i == 0 && j == 0)) {
                        if (cmpf(img.at<float>(x + i, y + j), img.at<float>(x, y)) &&
                            cmpf(val.at<float>(x + i, y + j), (val.at<float>(x, y) - 1.0)) &&
                            lab.at<float>(x + i, y + j) > 0 && lab.at<float>(x + i, y + j) < lmin) {
                            lmin = lab.at<float>(x + i, y + j);
                        }
                    }
                }
            }
        }
    }

    if (!cmpf(lmin, LMAX) && !cmpf(lab.at<float>(x, y), lmin)) {
        lab.at<float>(x, y) = lmin;
        changed = 1;
    }
}

/*! \brief Watersheed
 *
 *  Realiza a segmentação da imagem pelo algoritmo watersheed na variação rainfall
 *  Obs: este algoritmo foi retirado de uma tese de mestrado - Implementation of Watershed Based
 *  Image Segmentation Algorithm in FPGA, de Sameer Ruparelia.
 *
 *
 * \param Mat imagem em formato da matriz a ser manipulada
 * \param int comprimento da janela, valor default = 1
 * \param string Identifica a imagem na janela
 * \return Mat
 */
JNIEXPORT void JNICALL Java_com_example_bruno_seg_GalleryActivity_watershed(
        JNIEnv *env, jobject obj,
        jlong scrGray,
        jlong dst,
        jint sobelThreshold,
        jint bilateralSigmaColorSpace
){

    cv::Mat& srcGray = *((cv::Mat*)scrGray);
    cv::Mat& matDst = *((cv::Mat*)dst);
    new_label = 0.0;
    scan_step2 = 1;
    scan_step3 = 1;
    windowSize = 1;

    img=srcGray.clone();

    Mat bl;
    bilateralFilter ( img, bl, 15, bilateralSigmaColorSpace, bilateralSigmaColorSpace);

    img=bl.clone();

    img=sobel(img,sobelThreshold);
    img=morpho_op(img);

    img.convertTo(img, CV_32FC3);

    VMAX = 100000000000;
    LMAX = 100000000000;

    lab = Mat::zeros(img.rows,img.cols,CV_32FC3);
    val = Mat::zeros(img.rows,img.cols,CV_32FC3);

    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            lab.at<float>(x, y) = INIT;
            val.at<float>(x, y) = INIT;
        }
    }


    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            step1(x, y);
        }
    }

    while (scan_step2 == 1) {
        changed = 0;
        //scan top left >> bottom right
        for (int x = 0; x < img.rows; x++) {
            for (int y = 0; y < img.cols; y++) {
                step2(x, y);
            }
        }
        if (changed == 0) {
            scan_step2 = 0;
        } else {
            changed = 0;
            //scan bottom right >> top left
            for (int x = img.rows - 1; x >= 0; x--) {
                for (int y = img.cols - 1; y >= 0; y--) {
                    step2(x, y);
                }
            }
            if (changed == 0) {
                scan_step2 = 0;
            }
        }
    }
    int limite = 0;

    while (scan_step3 == 1) {
        limite++;
        if (limite == 10) break;
        changed = 0;
        // top left >> bottom right
        for (int x = 0; x < img.rows; x++) {
            for (int y = 0; y < img.cols; y++) {
                step3(x, y);
            }
        }

        if (changed == 0) {
            scan_step3 = 0;
        } else {
            changed = 0;
            //bottom right >> top left
            for (int x = img.rows - 1; x >= 0; x--) {
                for (int y = img.cols - 1; y >= 0; y--) {
                    step3(x, y);
                }
            }
            if (changed == 0) {
                scan_step3 = 0;
            }
        }
    }

    max_pixel = 0;

    for (int x = 0; x < lab.rows; x++) {
        for (int y = 0; y < lab.cols; y++) {
            if (max_pixel < lab.at<float>(x, y))
                max_pixel = lab.at<float>(x, y);
        }
    }

    lab.convertTo(lab, CV_8U, (255.0)/max_pixel);

    lab=color_watershed(lab);
    matDst=lab.clone();

}


JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_sobel(
        JNIEnv *env, jobject obj,
        jlong scrGray,
        jlong dst
){

    cv::Mat& matSrcGray = *((cv::Mat*)scrGray);
    cv::Mat& matDst = *((cv::Mat*)dst);

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
//    cv::Mat grad;//matDst
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    Sobel( matSrcGray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel( matSrcGray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    addWeighted( abs_grad_x, 1., abs_grad_y, 1., 0, matDst);
    for (int i = 0; i < matDst.rows; ++i) {
        for (int j = 0; j < matDst.cols; ++j) {
            if (matDst.at<uchar>(i,j) < MB_THRESHOLD)
                matDst.at<uchar>(i,j)=0;
        }
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
    for( unsigned int i = 0; i < v.size(); i++ ) {
        const cv::KeyPoint& kp = v[i];
        cv::circle(mRgb, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }
}






void JNICALL Java_com_example_bruno_seg_CameraManip_salt(JNIEnv *env, jobject instance,
                                                         jlong matAddrGray,
                                                         jint nbrElem,
                                                         jint nRows,
                                                         jint nCols
){
    cv::Mat &mGr=*(cv::Mat*) matAddrGray;
    for (int k = 0; k < nbrElem; ++k) {
        int i=rand() % nCols;
        int j=rand() % nRows;
        mGr.at<uchar>(j,i)=255;

    }
}


JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_toGray(JNIEnv *env, jobject obj,
                                                                     jlong src, jlong dst){
    cv::Mat &matSrc = *((cv::Mat*)src);
    cv::Mat &matDst = *((cv::Mat*)dst);
    cv::cvtColor(matSrc, matDst, CV_BGRA2GRAY);


}


JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_bilateralFilter(JNIEnv *env, jobject obj,
                                                                    jlong src, jlong dst){
    cv::Mat &matSrc = *((cv::Mat*)src);//You've got to use gray
    cv::Mat &matDst = *((cv::Mat*)dst);
    bilateralFilter ( matSrc, matDst, 15, 80, 80 );

}

JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_morphoOp(JNIEnv *env, jobject obj,
                                                                        jlong addrSrc, jlong addrDst){
    cv::Mat& mSrc  = *(cv::Mat*)addrSrc;
    cv::Mat& mDst = *(cv::Mat*)addrDst;
    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement( MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( mSrc, mDst, MORPH_CLOSE, element, Point(morph_size,morph_size), 3 );
}


Mat watershedOpencv(Mat src, Mat srcAux) {
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for( int x = 0; x < src.rows; x++ ) {
        for( int y = 0; y < src.cols; y++ ) {
            if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
                src.at<Vec3b>(x, y)[0] = 0;
                src.at<Vec3b>(x, y)[1] = 0;
                src.at<Vec3b>(x, y)[2] = 0;
            }
        }
    }
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) << 1,  1, 1,
                                    1, -8, 1,
                                    1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = srcAux; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    srcAux.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    srcAux = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    bw = srcAux.clone();
    //    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    return dst;
}



JNIEXPORT void JNICALL Java_com_example_bruno_seg_GalleryActivity_watershedOpencv(
        JNIEnv *env, jobject obj,
        jlong scrGray,
        jlong dst
) {

    cv::Mat &srcGray = *((cv::Mat *) scrGray);
    cv::Mat& matDst = *((cv::Mat*)dst);


    Mat src = srcGray.clone();
    cvtColor(src, src, CV_BGR2GRAY);
    Mat result = watershedOpencv(src, src);
    matDst=result.clone();

}
}


