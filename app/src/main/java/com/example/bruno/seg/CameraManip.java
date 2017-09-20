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
import org.opencv.imgproc.Imgproc;


public class CameraManip extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="SEG::Camera Manip";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_BILATERALFILTER = 2;
    private static final int       VIEW_MODE_MORPHOOP = 5;
    private static final int       VIEW_MODE_WATERSHED = 7;
    private static final int       VIEW_MODE_SOBEL = 6;
    private static final int       VIEW_MODE_SIFT = 9;

    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;

    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewBilateralFilter;
    private MenuItem               mItemPreviewMorphoOp;
    private MenuItem               mItemPreviewSobel;
    private MenuItem               mItemPreviewWatershed;
    private MenuItem               mItemPreviewSift;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after OpenCV initialization
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
        mItemPreviewRGBA = menu.add("RGBA");
        mItemPreviewGray = menu.add("GRAY C++");
        mItemPreviewSobel = menu.add("Sobel C++");
        mItemPreviewBilateralFilter = menu.add("Bilateral Filter C++");
        mItemPreviewMorphoOp = menu.add("Morphologic Operation C++");
        mItemPreviewSift = menu.add("Sift C++");
        mItemPreviewWatershed = menu.add("Watershed OpenCV");
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

            case VIEW_MODE_RGBA:
                // input frame has RBGA format
                mRgba = inputFrame.rgba();
                break;

            case VIEW_MODE_GRAY:
//                input frame has gray scale format
                toGray(inputFrame.rgba().getNativeObjAddr(),mRgba.getNativeObjAddr());
                break;

            case VIEW_MODE_SOBEL:
                mRgba = inputFrame.rgba();
                mGray=inputFrame.gray();
                sobel(mGray.getNativeObjAddr(),mRgba.getNativeObjAddr());
                break;

            case VIEW_MODE_BILATERALFILTER:
//                 input frame has gray scale format
                mGray=inputFrame.gray();
                bilateralFilter(mGray.getNativeObjAddr(),mRgba.getNativeObjAddr());
                break;

            case VIEW_MODE_MORPHOOP:
                // input frame has RGBA format
                mGray = inputFrame.gray();
                morphoOp(mGray.getNativeObjAddr(),mRgba.getNativeObjAddr());
                break;

            case VIEW_MODE_WATERSHED:
                mRgba=inputFrame.rgba();
                mRgba = watershed(mRgba);
                break;
            case VIEW_MODE_SIFT:
                FindFeatures(inputFrame.rgba().getNativeObjAddr(),mRgba.getNativeObjAddr());
                break;
        }

        return mRgba;
    }

    public Mat watershed(Mat mInput){
        Mat threeChannel = new Mat();
        Imgproc.cvtColor(mInput, mInput, Imgproc.COLOR_BGRA2BGR);
        Imgproc.cvtColor(mInput, threeChannel, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(threeChannel, threeChannel, 100, 255, Imgproc.THRESH_BINARY);

        Mat fg = new Mat(mInput.size(),CvType.CV_8U);
        Imgproc.erode(threeChannel,fg,new Mat(),new Point(-1,-1),2);

        Mat bg = new Mat(mInput.size(),CvType.CV_8U);
        Imgproc.dilate(threeChannel,bg,new Mat(),new Point(-1,-1),3);
        Imgproc.threshold(bg,bg,1, 128,Imgproc.THRESH_BINARY_INV);

        Mat markers = new Mat(mInput.size(),CvType.CV_8U, new Scalar(0));
        Core.add(fg, bg, markers);
        markers.convertTo(markers, CvType.CV_32S);
        Imgproc.watershed(mInput, markers);
        markers.convertTo(markers,CvType.CV_8U);

        return markers;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewBilateralFilter) {
            mViewMode = VIEW_MODE_BILATERALFILTER;
        } else if (item == mItemPreviewMorphoOp) {
            mViewMode = VIEW_MODE_MORPHOOP;
        } else if (item == mItemPreviewSobel) {
            mViewMode = VIEW_MODE_SOBEL;
        }else if (item == mItemPreviewWatershed) {
            mViewMode = VIEW_MODE_WATERSHED;
        } else if (item == mItemPreviewSift) {
            mViewMode = VIEW_MODE_SIFT;
        }
        return true;
    }

    /**
     * Native methods
     */

    public native void bilateralFilter(long matAddrSrc,long matAddDst);
    public native void toGray(long matSrc, long matDst);
    public native void sobel(long matAddrSrc,long matAddrDst);
    public native void morphoOp(long matSrc, long matDst);
    public native void FindFeatures(long addrGray,long addrRgba);

    public native void salt(long matAddrSrc, int nbrElem,int mRows, int mCols);
    public native void thresh(long matAddrRgba,long matDst);


}
