package com.example.bruno.seg;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class GalleryActivity extends AppCompatActivity {
    private static final String TAG = "SEG::Gallery Activity";
    public static final int IMAGE_GALLERY_REQUEST = 20;
    private SeekBar seekBarSobel = null;
    private SeekBar seekBarBilateral = null;
    int progressChanged = 20;
    int progressChangedBilateral = 80;
    public ImageView imgPicture;
    public static Bitmap lastBitmap;
    public Mat mMat;
    public Mat mMatDst;
    public Mat mGray;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d("Gallery Activity", "OpenCV not initialized");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);

        seekBarSobel = (SeekBar) findViewById(R.id.seekBarSobel);
        seekBarBilateral = (SeekBar) findViewById(R.id.seekBarBilateral);
        imgPicture = (ImageView) findViewById(R.id.imgPicture);

        seekBarSobel.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                progressChanged = progress;
            }

            public void onStartTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }

            public void onStopTrackingTouch(SeekBar seekBar) {
                Toast.makeText(GalleryActivity.this, "Sobel seek bar progress:" + progressChanged,
                        Toast.LENGTH_SHORT).show();
            }
        });

        seekBarBilateral.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                progressChangedBilateral = progress;
            }

            public void onStartTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }

            public void onStopTrackingTouch(SeekBar seekBar) {
                Toast.makeText(GalleryActivity.this, "Bilateral seek bar progress:" + progressChangedBilateral,
                        Toast.LENGTH_SHORT).show();
            }
        });
    }

    public void onImageGalleryClicked(View v) {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        File pictureDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        String pictureDirectoryPath = pictureDirectory.getPath();
        Uri data = Uri.parse(pictureDirectoryPath);
        photoPickerIntent.setDataAndType(data, "image/*");

        startActivityForResult(photoPickerIntent, IMAGE_GALLERY_REQUEST);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("GalleryActivity::OpenCV", "OpenCV loaded successfully");
                    System.loadLibrary("native-lib");
                    System.loadLibrary("opencv_java3");

                    mMat = new Mat();
                    mMatDst = new Mat();
                    mGray = new Mat();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            // if we are here, everything processed successfully.
            if (requestCode == IMAGE_GALLERY_REQUEST) {
                // if we are here, we are hearing back from the image gallery.
                // the address of the image on the SD Card.
                Uri imageUri = data.getData();

                // declare a stream to read the image data from the SD Card.
                InputStream inputStream;

                // we are getting an input stream, based on the URI of the image.
                try {
                    inputStream = getContentResolver().openInputStream(imageUri);
                    Bitmap image = BitmapFactory.decodeStream(inputStream);
                    lastBitmap = Bitmap.createBitmap(image);

                    // show the image to the user
                    imgPicture.setImageBitmap(image);

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    // show a message to the user indictating that the image is unavailable.
                    Toast.makeText(this, "Unable to open image", Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    //    TODO: linhas / coluna
    public void ClickWatershed(View v) {
//        new MyTask().execute("lastBitmap");
        if (lastBitmap != null) {
            mMat = new Mat();
            mMatDst = new Mat();

            Utils.bitmapToMat(lastBitmap, mMat);
            Imgproc.cvtColor(mMat, mMat, Imgproc.COLOR_RGBA2GRAY);

            watershed(mMat.getNativeObjAddr(), mMatDst.getNativeObjAddr(), progressChanged, progressChangedBilateral);
            letsSeeMat(mMatDst);

//            bm = Bitmap.createBitmap(mMatDst.cols(), mMatDst.rows(),Bitmap.Config.ARGB_8888);//consegue, mas da problema de alocacao de memoria
            letsSeeBitmap(lastBitmap);
            Utils.matToBitmap(mMatDst, lastBitmap);

            imgPicture.setImageBitmap(lastBitmap);


        }
    }

    public void ClickWatershedJava(View v) {

        if (lastBitmap != null) {
            mMat = new Mat();
            mMatDst = new Mat();

            Utils.bitmapToMat(lastBitmap, mMat);

            mMatDst = watershedJava(mMat);

            Utils.matToBitmap(mMatDst, lastBitmap);

            imgPicture.setImageBitmap(lastBitmap);

        }

    }

    public Mat watershedJava(Mat mInput) {
        Mat threeChannel = new Mat();
        Imgproc.cvtColor(mInput, mInput, Imgproc.COLOR_BGRA2BGR);
        Imgproc.cvtColor(mInput, threeChannel, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(threeChannel, threeChannel, 100, 255, Imgproc.THRESH_BINARY);

        Mat fg = new Mat(mInput.size(), CvType.CV_8U);
        Imgproc.erode(threeChannel, fg, new Mat(), new Point(-1, -1), 2);

        Mat bg = new Mat(mInput.size(), CvType.CV_8U);
        Imgproc.dilate(threeChannel, bg, new Mat(), new Point(-1, -1), 3);
        Imgproc.threshold(bg, bg, 1, 128, Imgproc.THRESH_BINARY_INV);

        Mat markers = new Mat(mInput.size(), CvType.CV_8U, new Scalar(0));
        Core.add(fg, bg, markers);
        markers.convertTo(markers, CvType.CV_32S);
        Imgproc.watershed(mInput, markers);
        markers.convertTo(markers, CvType.CV_8U);

        return markers;
    }

    private void letsSeeMat(Mat pMat) {
        Log.i("Lets see: ", "Mat " +
                "\nHeight -> " + pMat.height()+
                "\nWidth -> " + pMat.width() +
                "\nrows -> " + pMat.rows() +
                "\ncols -> " + pMat.cols()
        );


    }

    private void letsSeeBitmap(Bitmap bmp) {

            Log.i("LETSSEE",bmp.getConfig().name());
            Log.i("Bitmap - ","Lets see lastBitmap"+
                    "\ngetHeight -> "+ String.valueOf(bmp.getHeight())+
                    "\ngetWidth -> "+ String.valueOf(bmp.getWidth())+
                    "\nconfig -> "+ String.valueOf(bmp.getConfig())
            );
    }

    private class MyTask extends AsyncTask<String, Integer, String> {
        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected String doInBackground(String... strings) {
            String str = strings[0];

            if(lastBitmap!= null  ) {
                mMat=new Mat();
                mMatDst=new Mat();

                Utils.bitmapToMat(lastBitmap, mMat);
                Imgproc.cvtColor(mMat,mMat,Imgproc.COLOR_RGBA2GRAY);

                watershed(mMat.getNativeObjAddr(),mMatDst.getNativeObjAddr(),progressChanged,progressChangedBilateral);

                Utils.matToBitmap(mMatDst,lastBitmap);

            }

            return str;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            imgPicture.setImageBitmap(lastBitmap);
        }
    }

    public native void watershed(long addrGray,long addrRgba, int sobelThreshold,int bilateralParameter);



}



