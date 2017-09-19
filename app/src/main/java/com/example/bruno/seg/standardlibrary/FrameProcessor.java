package com.example.bruno.seg.standardlibrary;

public interface FrameProcessor
{
	public void doFrameProcessing(byte[] pFrame, int pWidth, int pHeight);
	
	public String getDescription();
}
