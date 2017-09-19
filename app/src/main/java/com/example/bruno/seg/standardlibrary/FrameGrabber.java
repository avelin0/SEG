package com.example.bruno.seg.standardlibrary;

import java.io.IOException;
import android.app.Activity;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PreviewCallback;
import android.os.Build;
import android.view.SurfaceHolder;
import android.view.SurfaceHolder.Callback;
import android.view.ViewGroup.LayoutParams;
import android.view.SurfaceView;

public abstract class FrameGrabber implements Callback, PreviewCallback
{
	protected int mWidth;
	
	protected int mHeight;

	protected int mCameraId;

	protected Camera mCamera;
	
	protected SurfaceView mBaseSurface;
	
	protected FrameGrabber(Context pContext, int pCameraId, int pWidth, int pHeight)
	{
		mWidth = pWidth;
		
		mHeight = pHeight;

		mCameraId = pCameraId;

		mBaseSurface = new SurfaceView(pContext);
		
		mBaseSurface.getHolder().addCallback(this);
	}

	public abstract void onPreviewFrame(byte[] pFrame, Camera pCamera);
	
	@Override
	public void surfaceChanged(SurfaceHolder pHolder, int pFormat, int pWidth, int pHeight)
	{
		setupCamera();
	}

	@Override
	public void surfaceCreated(SurfaceHolder pHolder)
	{
		connectCamera();
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder pHolder)
	{
		releaseCamera();
	}	
	
	private void releaseCamera()
	{
		mCamera.setPreviewCallback(null);
		
		mCamera.stopPreview();
		
		mCamera.release();
		
		mCamera = null;
	}
	
	private void setupCamera()
	{
		Parameters parameters = mCamera.getParameters();
		
		parameters.setPreviewSize(mWidth, mHeight);
		
		parameters.setPreviewFormat(ImageFormat.NV21);
		
		mCamera.setParameters(parameters);
		
		mCamera.setPreviewCallback(this);
		
		mCamera.startPreview();
	}
	
	private void connectCamera()
	{
		if ( Camera.getNumberOfCameras() > 1 )
		{
			mCamera = Camera.open(mCameraId);
		}
		
		else
		{
			mCamera = Camera.open();
		}
		
		try
		{
			// Android API < 3.0 (HONEYCOMB)
			if ( Build.VERSION.SDK_INT < 11 )
			{
				mCamera.setPreviewDisplay(null);
			}
			
			// Android API >= 3.0 (HONEYCOMB)
			else
			{
				mCamera.setPreviewDisplay(mBaseSurface.getHolder());
			}

			mCamera.setPreviewCallback(this);
		}
		
		catch ( IOException e )
		{
			mCamera.release();
			
			mCamera = null;
		}
	}
	
	public void onPause()
	{
		mCamera.stopPreview();
	}
	
	public void reparentTo(Activity pActivity)
	{
		pActivity.addContentView(mBaseSurface, new LayoutParams(1, 1));
	}
	
	public void changeSourceCamera()
	{
		int numCamereas = Camera.getNumberOfCameras();
		
		if ( numCamereas > 1 )
		{
			mCameraId = ( mCameraId + 1 ) % numCamereas;
			
			releaseCamera();
			
			connectCamera();
			
			setupCamera();
		}
	}
	
	public int getCameraId()
	{
		return mCameraId;
	}
	
	public int getWidth()
	{
		return mWidth;
	}
	
	public int getHeight()
	{
		return mHeight;
	}
}
