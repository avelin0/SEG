package com.example.bruno.seg.standardlibrary;

import android.graphics.Bitmap;

public abstract class GlColorSpace
{
	static
	{
    	System.loadLibrary("dolphin");
	}
	
	public static native void toRGB(Bitmap out, byte[] pFrame, int pWidth, int pHeight);
}
