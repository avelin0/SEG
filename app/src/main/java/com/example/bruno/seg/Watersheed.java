package com.example.bruno.seg;

import org.opencv.core.Mat;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC3;

/**
 * Created by bruno on 16/09/17.
 */

public class Watersheed {
    public Mat mMat;

    private Mat lab;
    private Mat val;
    private Mat img;
    private int show=0;
    private int save=0;


//Watersheed
    private const float INIT = 0.0;
    private float max_pixel;
    private float VMAX;
    private float LMAX;
    private float new_label = 0.0;
    private int changed = 0;
    private int scan_step2 = 1;
    private int scan_step3 = 1;
    private int windowSize = 1;

    public Watershed() {

    }

    private boolean cmpf(float a, float b, float epsilon = 0.005f) {
        return (Math.abs(a - b) < epsilon);
    }

    public Mat doWatershed(Mat imagem, int winSize=1) {
            new_label = 0.0;
            scan_step2 = 1;
            scan_step3 = 1;
            windowSize = winSize;
            img = imagem;

            img.convertTo(img, CV_32FC3);

            VMAX = 100000000000;
            LMAX = 100000000000;

            //inicializar o tamanho da matriz
            lab = img.clone();
            val = img.clone();

            //inicializar os valores
            for (int x = 0; x < img.rows(); x++) {
                for (int y = 0; y < img.cols(); y++) {
                    lab.at<float>(x, y) = INIT;
                    val.at<float>(x, y) = INIT;
                }
            }
            //encontra menor pixel de cada vizinhanca
            for (int x = 0; x < img.rows(); x++) {
                for (int y = 0; y < img.cols(); y++) {
                    step1(x, y);
                }
            }

            //encontrar os plateaus mesmo pixel greyscale a partir dos minimos
            while (scan_step2 == 1) {
                changed = 0;
                //scan top left >> bottom right
                for (int x = 0; x < img.rows(); x++) {
                    for (int y = 0; y < img.cols(); y++) {
                        step2(x, y);
                    }
                }
                if (changed == 0) {
                    scan_step2 = 0;
                } else {
                    changed = 0;
                    //scan bottom right >> top left
                    for (int x = img.rows() - 1; x >= 0; x--) {
                        for (int y = img.cols() - 1; y >= 0; y--) {
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
                for (int x = 0; x < img.rows(); x++) {
                    for (int y = 0; y < img.cols(); y++) {
                        step3(x, y);
                    }
                }

                if (changed == 0) {
                    scan_step3 = 0;
                } else {
                    changed = 0;
                    //scan bottom right >> top left
                    for (int x = img.rows() - 1; x >= 0; x--) {
                        for (int y = img.cols() - 1; y >= 0; y--) {
                            step3(x, y);
                        }
                    }
                    if (changed == 0) {
                        scan_step3 = 0;
                    }
                }
            }

            max_pixel = 0;
            for (int x = 0; x < lab.rows(); x++) {
                for (int y = 0; y < lab.cols(); y++) {
                    if (max_pixel < lab.at<float>(x, y)) max_pixel = lab.at<float>(x, y);
                }
            }

            lab.convertTo(lab, CV_8U, 255.0 / (max_pixel));

            return lab;
        }
    }

    public Mat color_watershed(Mat color) {
        // Gera cores aleatorias
//        TODO: verificar como acesso Vec3b no java
        vector<Vec3b> colors;
//        TODO: size_t?
        for (size_t i = 0; i < 255; i++) {
//            verificar theRNG no java
            int b = theRNG().uniform(10, 255);
            int g = theRNG().uniform(10, 255);
            int r = theRNG().uniform(10, 255);
//            TODO: verificar uso de vector push_back no java
//            TODO: verificar uso de uchar no java
            colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
        }
        // Cria imagem final
        Mat dst = Mat::zeros(color.size(), CV_8UC3);

        // Pinta cada area de uma cor
        for (int i = 0; i < color.rows(); i++) {
            for (int j = 0; j < color.cols(); j++) {

                color.convertTo(color, CV_32F);
//                TODO: verificar acesso a cada pixel no java ( uso do at)
                int index = color.at<int>(i, j) % (int) 255;
                if (index > 0 && index <= 255) {
                    dst.at<Vec3b>(i, j) = colors[index - 1];
                } else
                    dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }

        }
        return dst;
    }

    private void step3(int x, int y) {
        float lmin = LMAX;
        float fmin = img.at<float>(x, y);
        if (cmpf(val.at<float>(x, y), 0.0)) {
            for (int i = -windowSize; i <= windowSize; i++) {
                for (int j = -windowSize; j <= windowSize; j++) {
                    if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1 && !(i == 0 && j == 0)) {
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
                        if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1 &&
                                !(i == 0 && j == 0)) {
                            if (img.at<float>(x + i, y + j) < fmin) {
                                fmin = img.at<float>(x + i, y + j);
                            }
                        }
                    }
                }

                for (int i = -windowSize; i <= windowSize; i++) {
                    for (int j = -windowSize; j <= windowSize; j++) {
                        if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1 &&
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
                        if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1 &&
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

    private void step2(int x, int y) {
        if (val.at<float>(x, y) != 1) {
            float min = VMAX;
            for (int i = -windowSize; i <= windowSize; i++) {
                for (int j = -windowSize; j <= windowSize; j++) {
                    if (!(i == 0 && j == 0)) {
                        if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1) {
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

    private void step1(int x, int y) {
        if (val.at<float>(x, y) != 1) {
            //inspecionar os vizinhos do pixel para setar o marcador
            for (int i =- windowSize; i <= windowSize; i++) {
                for (int j =- windowSize; j <= windowSize; j++) {
                    //Casos de borda
                    if (!(i == 0 && j == 0)) {
                        if (x + i >= 0 && y + j >= 0 && x + i <= img.rows() - 1 && y + j <= img.cols() - 1) {

                            if (img.at<float>(x, y) > img.at<float>(x + i, y + j)) {
                                val.at<float>(x, y) = 1.0;
                            }
                        }
                    }
                }
            }
        }
    }
}
