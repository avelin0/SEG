package framework;

import java.util.Observable;
import java.util.Observer;

import com.example.bruno.seg.standardlibrary.FrameContainer;
import com.example.bruno.seg.standardlibrary.FrameProcessor;

public class FrameProcessorWorker extends Observable implements FrameContainer, Observer, Runnable
{
	private byte[] mFrame;
	
	private int mFrameWidth;
	
	private int mFrameHeight;
	
	private Object mFrameLock;
	
	private boolean mFrameFlag;
	
	private FrameProcessor mFrameProcessor;
	
	public FrameProcessorWorker()
	{
		super();
		
		mFrameFlag = false;
		
		mFrameLock = new Object();
	}
	
	@Override
	public void run()
	{
		while ( true )
		{	
			if ( mFrameFlag )
			{
				synchronized ( mFrameLock )
				{
					if ( mFrameProcessor != null )
					{
						mFrameProcessor.doFrameProcessing(mFrame, mFrameWidth, mFrameHeight);
					}
					
					setChanged();
					
					notifyObservers();
					
					mFrameFlag = false;
					
					try
					{
						mFrameLock.wait();
					}
					
					catch (InterruptedException e)
					{
						e.printStackTrace();
					}
				}
			}
		}
	}

	@Override
	public void update(Observable observable, Object data)
	{
		synchronized ( mFrameLock )
		{
			mFrame = ((FrameContainer) observable).getFrame();
			
			mFrameWidth = ((FrameContainer) observable).getFrameWidth();
			
			mFrameHeight = ((FrameContainer) observable).getFrameHeight();
			
			mFrameFlag = true;
			
			mFrameLock.notify();
		}
	}
	
	@Override
	public int getFrameWidth()
	{
		return mFrameWidth;
	}
	
	@Override
	public int getFrameHeight()
	{
		return mFrameHeight;
	}
	
	@Override
	public byte[] getFrame()
	{
		return mFrame;
	}
	
	public void setFrameProcessor(FrameProcessor pFrameProcessor)
	{
		mFrameProcessor = pFrameProcessor;
	}
	
	public String getInfo()
	{
		return mFrameProcessor.getDescription();
	}
}
