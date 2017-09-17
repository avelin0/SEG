#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <android/bitmap.h>
#include <android/log.h>

extern "C"{

#define MB_THRESHOLD 20

using namespace std;
using namespace cv;

Mat lab;
Mat val;
Mat img;
int show=0;
int save=0;

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
        //inspecionar os vizinhos do pixel para setar o marcador
        for (int i =- windowSize; i <= windowSize; i++) {
            for (int j =- windowSize; j <= windowSize; j++) {
                //Casos de borda
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
                        //min = maior vizinho
                        //encontra o maior valor vizinho de minimo
                        if (img.at<float>(x, y) == img.at<float>(x + i, y + j) && val.at<float>(x + i, y + j) > 0 &&
                            val.at<float>(x + i, y + j) < min) {
                            min = val.at<float>(x + i, y + j);
                        }
                    }
                }
            }
        }
        //novo valor do pixel baseado nos minimos vizinhos
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
 *
 * \param Mat imagem em formato da matriz a ser manipulada
 * \param int comprimento da janela, valor default = 1
 * \param string Identifica a imagem na janela
 * \return Mat
 */
//Mat Watershed(Mat imagem, int winSize=1,string mensagem="Watersheed") {
JNIEXPORT void JNICALL Java_com_example_bruno_seg_GalleryActivity_watershed(
        JNIEnv *env, jobject obj,
        jlong scrGray,
        jlong dst
){

    cv::Mat& img = *((cv::Mat*)scrGray);
    cv::Mat& lab = *((cv::Mat*)dst);

    new_label = 0.0;
    scan_step2 = 1;
    scan_step3 = 1;
    windowSize = 1;
//    img = imagem;

    img.convertTo(img, CV_32FC3);

    VMAX = 100000000000;
    LMAX = 100000000000;

    //inicializar o tamanho da matriz
    lab = img.clone();
    val = img.clone();

    //inicializar os valores
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            lab.at<float>(x, y) = INIT;
            val.at<float>(x, y) = INIT;
        }
    }
    //encontra menor pixel de cada vizinhanca
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            step1(x, y);
        }
    }

    //encontrar os plateaus mesmo pixel greyscale a partir dos minimos
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
        //scan top left >> bottom right
        for (int x = 0; x < img.rows; x++) {
            for (int y = 0; y < img.cols; y++) {
                step3(x, y);
            }
        }

        if (changed == 0) {
            scan_step3 = 0;
        } else {
            changed = 0;
            //scan bottom right >> top left
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
            if (max_pixel < lab.at<float>(x, y)) max_pixel = lab.at<float>(x, y);
        }
    }

    lab.convertTo(lab, CV_8U, 255.0 / (max_pixel));
//    lab=color_watershed(lab);
//    return lab;
}


/*! \brief Watersheed
 *
 *  Realiza a segmentação da imagem pelo algoritmo watersheed na variação rainfall
 *
 * \param Mat imagem em formato da matriz a ser manipulada
 * \param int comprimento da janela, valor default = 1
 * \param string Identifica a imagem na janela
 * \return Mat
 */
JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_watershed(
        JNIEnv *env, jobject obj,
        jlong scrGray,
        jlong dst
){

    cv::Mat& img = *((cv::Mat*)scrGray);
    cv::Mat& lab = *((cv::Mat*)dst);
//Mat Watershed(Mat imagem)

    int winSize=1;
    new_label = 0.0;
    scan_step2 = 1;
    scan_step3 = 1;
    windowSize = winSize;
//    img = imagem;

    img.convertTo(img, CV_32FC3);

    VMAX = 100000000000;
    LMAX = 100000000000;

    //inicializar o tamanho da matriz
    lab = img.clone();
    val = img.clone();
    //inicializar os valores
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            lab.at<float>(x, y) = INIT;
            val.at<float>(x, y) = INIT;
        }
    }
    //encontra menor pixel de cada vizinhanca
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            step1(x, y);
        }
    }



    //encontrar os plateaus mesmo pixel greyscale a partir dos minimos
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
    __android_log_print(ANDROID_LOG_INFO, "Native-Lib", "Aqui 1", 1);
    int limite = 0;
    while (scan_step3 == 1) {
        limite++;
        if (limite == 10) break;
        changed = 0;
        //scan top left >> bottom right
        for (int x = 0; x < img.rows; x++) {
//            TODO: aqui da crash
            for (int y = 0; y < img.cols; y++) {
                step3(x, y);
            }
        }

        if (changed == 0) {
            scan_step3 = 0;
        } else {
            changed = 0;
            //scan bottom right >> top left
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
            if (max_pixel < lab.at<float>(x, y)) max_pixel = lab.at<float>(x, y);
        }
    }

    lab.convertTo(lab, CV_8U, 255.0 / (max_pixel));
//    lab=color_watershed(lab);

//    TODO: problema? sera??
//    matDst=lab;
//    return lab;
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

JNIEXPORT void JNICALL Java_com_example_bruno_seg_GalleryActivity_FindFeatures(JNIEnv*, jobject,
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


void printNative(){
    __android_log_print(ANDROID_LOG_VERBOSE, "Native-Lib", "teste", 1);
}

//TODO: nr Colunas e Linhas nao e o mesmo
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
//        printf("Linhas: %d , Colunas: %d",i,j);
        printNative();

    }
}


JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_toGray(JNIEnv *env, jobject obj,
                                                                     jlong src, jlong dst){
    cv::Mat &matSrc = *((cv::Mat*)src);
    cv::Mat &matDst = *((cv::Mat*)dst);
    cv::cvtColor(matSrc, matDst, CV_BGRA2GRAY);


}

JNIEXPORT void JNICALL Java_com_example_bruno_seg_GalleryActivity_toGray(JNIEnv *env, jobject obj,
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

//TODO: analisar melhos os parametros, but it is working. Obs: implementacao aqui esta diferente da do C++
JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_morphoOp(JNIEnv *env, jobject obj,
                                                                        jlong addrSrc, jlong addrDst){
    cv::Mat& mSrc  = *(cv::Mat*)addrSrc;
    cv::Mat& mDst = *(cv::Mat*)addrDst;
    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement( MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( mSrc, mDst, MORPH_CLOSE, element, Point(morph_size,morph_size), 3 );
}

//TODO:not working
JNIEXPORT void JNICALL Java_com_example_bruno_seg_CameraManip_thresh(JNIEnv *env, jobject obj,
                                                                     jlong scrGray,
                                                                     jlong dst,
                                                                     double thresholdValue=50.0,
                                                                     double maxVal=100.0){
    cv::Mat& matSrcGray = *((cv::Mat*)scrGray);
    cv::Mat& matDst = *((cv::Mat*)dst);
    threshold( matSrcGray, matDst, thresholdValue, maxVal, CV_THRESH_BINARY);
}



}


