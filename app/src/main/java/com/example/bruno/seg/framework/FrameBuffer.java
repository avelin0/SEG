package framework;

import java.util.Observable;

import com.example.bruno.seg.standardlibrary.FrameContainer;
import com.example.bruno.seg.standardlibrary.FrameListener;

public class FrameBuffer extends Observable implements FrameListener, FrameContainer
{
	private int mFrameWidth;
	
	private int mFrameHeight;
	
	private byte[] mFrame;
	
	public FrameBuffer()
	{
		super();
	}
	
	@Override
	public void setNewFrame(byte[] pFrame, int pWidth, int pHeight)
	{
		mFrame = pFrame;
		
		mFrameWidth = pWidth;
		
		mFrameHeight = pHeight;

		setChanged();
		
		notifyObservers();
	}
	
	@Override
	public byte[] getFrame()
	{
		return mFrame;
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
}
