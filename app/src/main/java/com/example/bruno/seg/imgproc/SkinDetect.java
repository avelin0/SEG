package com.example.bruno.seg.imgproc;

public abstract class SkinDetect
{
	public static native void skinDetector1(byte[] pFrame, int pWidth, int pHeight);
	
	public static native void skinDetector2(byte[] pFrame, int pWidth, int pHeight);
}
