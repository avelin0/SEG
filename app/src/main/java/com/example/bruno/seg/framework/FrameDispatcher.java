package framework;

import java.util.ArrayList;
import java.util.List;

import com.example.bruno.seg.standardlibrary.FrameGrabber;
import com.example.bruno.seg.standardlibrary.FrameListener;

import android.content.Context;
import android.hardware.Camera;

public class FrameDispatcher extends FrameGrabber 
{
	private List<FrameListener> mFrameListeners;
	
	public FrameDispatcher(Context pContext, int pCameraId, int pWidth, int pHeight)
	{
		super(pContext, pCameraId, pWidth, pHeight);
		
		mFrameListeners = new ArrayList<FrameListener>();
	}
	
	public void addFrameListener(FrameListener pFrameListener)
	{
		mFrameListeners.add(pFrameListener);
	}

	@Override
	public void onPreviewFrame(byte[] pFrame, Camera pCamera)
	{
		for ( FrameListener framelistener : mFrameListeners )
		{
			framelistener.setNewFrame(pFrame, mWidth, mHeight);
		}
	}
}
