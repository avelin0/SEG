package com.example.bruno.seg;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;


public class CameraManip extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="OCV Sample::Activity";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int       VIEW_MODE_FEATURES = 5;
    private static final int       VIEW_MODE_WATERSHEED = 7;
    private static final int       VIEW_MODE_SOBEL = 6;

    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;

    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewCanny;
    private MenuItem               mItemPreviewFeatures;
    private MenuItem               mItemPreviewSobel;
    private MenuItem               mItemPreviewWatersheed;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
//                    System.loadLibrary("mixed_sample");
                    System.loadLibrary("native-lib");
                    System.loadLibrary("opencv_java3");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public void Tutorial2Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }


    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//        System.loadLibrary("opencv_java3");
//    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.camera_manip);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_surface);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewSobel = menu.add("Sobel");
        mItemPreviewFeatures = menu.add("Find features (C++)");
        mItemPreviewWatersheed = menu.add("Watersheed OpenCV");
        return true;
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
            case VIEW_MODE_GRAY:
                // input frame has gray scale format
                Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_RGBA:
                // input frame has RBGA format
                mRgba = inputFrame.rgba();
                break;
            case VIEW_MODE_CANNY:
//                 input frame has gray scale format
                mRgba = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mIntermediateMat, 80, 100);
                Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case VIEW_MODE_SOBEL:
                mRgba = inputFrame.rgba();
                Imgproc.Sobel(inputFrame.gray(),mRgba,CvType.CV_16S,1,1,3,1,0);
                break;
            case VIEW_MODE_FEATURES:
                // input frame has RGBA format
                mRgba = inputFrame.rgba();
                mGray = inputFrame.gray();

                FindFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
                break;
            case VIEW_MODE_WATERSHEED:

//                Mat bg = new Mat(mRgba.size(), CvType.CV_8U);
//                Mat threeChannel = new Mat();
//                Mat mRgb = new Mat();
//                Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_BGRA2BGR);
//
//                Imgproc.cvtColor(mRgba, threeChannel, Imgproc.COLOR_BGR2GRAY);
//                Imgproc.threshold(threeChannel, threeChannel, 100, 255, Imgproc.THRESH_BINARY);
//
//                Mat fg = new Mat(mRgb.size(), CvType.CV_8U);
//                Imgproc.erode(threeChannel, fg, new Mat(), new Point(-1, -1), 2);
//                Imgproc.dilate(threeChannel, bg, new Mat(), new Point(-1, -1), 3);
//                Imgproc.threshold(bg, bg, 1, 128, Imgproc.THRESH_BINARY_INV);
//                Imgproc.threshold(mRgba,mRgba,1,128,Imgproc.THRESH_BINARY);
//                Imgproc.

//                Mat markers = new Mat(mRgb.size(), CvType.CV_8U, new Scalar(0));
//                Core.add(fg, bg, markers);
//                mRgba.convertTo(mRgba,CvType.CV_8UC3);
//                markers.convertTo(markers, CvType.CV_32SC1);
//                Imgproc.watershed(mRgba, markers);
//                markers.convertTo(mRgba, CvType.CV_8U);
//                Imgproc.
//                Mat inter = new Mat();
//                Imgproc.cvtColor(mRgba,inter,Imgproc.COLOR_BGRA2BGR);
//                inter=steptowatershed(inter);
//                inter.copyTo(mRgba);
//                Imgproc.cvtColor(inter,mRgba,Imgproc.COLOR_BGR2BGRA);
//                to_gray(mRgba.getNativeObjAddr());
//                salt(mRgba.getNativeObjAddr(),1000000);
                break;
        }

        return mRgba;
    }

    public Mat steptowatershed(Mat img)
    {
        Mat threeChannel = new Mat();

        Imgproc.cvtColor(img, threeChannel, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(threeChannel, threeChannel, 100, 255, Imgproc.THRESH_BINARY);

        Mat fg = new Mat(img.size(),CvType.CV_8U);
        Imgproc.erode(threeChannel,fg,new Mat());

        Mat bg = new Mat(img.size(),CvType.CV_8U);
        Imgproc.dilate(threeChannel,bg,new Mat());
        Imgproc.threshold(bg,bg,1, 128,Imgproc.THRESH_BINARY_INV);

        Mat markers = new Mat(img.size(),CvType.CV_8U, new Scalar(0));
        Core.add(fg, bg, markers);
        Mat result1;
        WatershedSegmenter segmenter = new WatershedSegmenter();
        segmenter.setMarkers(markers);
        result1 = segmenter.process(img);
        return result1;
    }

    public class WatershedSegmenter {
        public Mat markers=new Mat();

        public void setMarkers(Mat markerImage)
        {

            markerImage.convertTo(markers, CvType.CV_32SC1);
        }

        public Mat process(Mat image)
        {
            Imgproc.watershed(image,markers);
            markers.convertTo(markers,CvType.CV_8U);
            return markers;
        }
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_CANNY;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_FEATURES;
        } else if (item == mItemPreviewSobel) {
            mViewMode = VIEW_MODE_SOBEL;
        }else if (item == mItemPreviewWatersheed) {
            mViewMode = VIEW_MODE_WATERSHEED;
        }

        return true;
    }



    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native void FindFeatures(long matAddrGr, long matAddrRgba);
    public native void thresh(long matAddrRgba,long matDst);
    public native void salt(long matAddrGray, int nbrElem);
    public native void morphoOp(long matSrc, long matDst);
//    public native void to_gray(long matAddr);

}
