package com.example.bruno.seg.imgproc;

public abstract class FrameDiff
{
	public static native void frameDiff(byte[] pFrame1, byte[] pFrame2, int pWidth, int pHeight);
}
