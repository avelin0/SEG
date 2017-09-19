package framework;

import com.example.bruno.seg.standardlibrary.GlColorSpace;
import com.example.bruno.seg.standardlibrary.FrameContainer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Observable;
import java.util.Observer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.opengl.GLU;
import android.opengl.GLUtils;
import android.opengl.GLSurfaceView.Renderer;
import android.util.DisplayMetrics;
import android.util.Log;

@SuppressLint("ViewConstructor")
public class FramePreviewSurface extends GLSurfaceView implements Observer, Renderer
{
	public static final int COLOR_ARGB_8888 = 0;
	
	public static final int COLOR_RGB_565 = 1;
	
	public static final int COLOR_GRAY = 2;
	
	private int mWidth;
	
	private int mHeight;
	
	private byte[] mFrame;
	
	private Object mFrameLock = new Object();
	
	private boolean mFrameFlag = false;
	
	private boolean mInvertH;
	
	private Bitmap mTexture;
	
	private int[] mTextureId;
	
	private float[] mVertexCoords;
	
	private float[] mTextureCoords;
	
	private FloatBuffer mTextureBuffer;	
	
	private FloatBuffer mVertexBuffer;
	
	private int mTextureSize;
	
	private int mTextureType = COLOR_RGB_565;
	
	private float[] mEye;
	
	private float[] mCenter;
	
	private float[] mUp;
	
	private float mNear;
	
	private float mFar;
	
	private float mRatio;
	
	private float mLeft;
	
	private float mRight;
	
	private float mTop;
	
	private float mBottom;
	
	private int mNumFrames = 0;
	
	private long mFpsStart = 0;
	
	public FramePreviewSurface(Context pContext, int pWidth, int pHeight)
	{
		super(pContext);
		
		setEGLConfigChooser(8, 8, 8, 8, 16, 0);
		
		getHolder().setFormat(PixelFormat.TRANSLUCENT);
		
		setRenderer(this);
		
		DisplayMetrics metrics = pContext.getResources().getDisplayMetrics();
		
		mRatio =  (float) metrics.heightPixels / metrics.widthPixels;

		mEye = new float[] { 0.0f, 0.0f, 0.5f };
		
		mCenter = new float[] { 0.0f, 0.0f, 0.0f };
		
		mUp = new float[] { 0.0f, -1.0f, 0.0f };
		
		mNear = 0.5f;
		
		mFar = 100.0f;
		
		mRight = (float) (1 / Math.pow(1 + mRatio * mRatio, 3));
		
		mLeft = -mRight;
		
		mTop = (float) (mRatio / Math.pow(1 + mRatio * mRatio, 3));
		
		mBottom = -mTop;
		
		mWidth = pWidth;
		
		mHeight = pHeight;
		
		mInvertH = false;
		
		build();
	}

	@Override
	public void update(Observable observable, Object data)
	{
		synchronized ( mFrameLock )
		{
			mFrame  = ((FrameContainer) observable).getFrame();
			mWidth  = ((FrameContainer) observable).getFrameWidth(); 
			mHeight = ((FrameContainer) observable).getFrameHeight();
			
			mFrameFlag = true;
		}
	}

	public void updateFrame(byte[] pFrame, int pWidth, int pHeight)
	{
		synchronized ( mFrameLock )
		{
			mFrame  = pFrame;
			mWidth  = pWidth; 
			mHeight = pHeight;
			
			mFrameFlag = true;
		}
	}

	@Override
	public void onDrawFrame(GL10 gl)
	{
		gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
		
		gl.glMatrixMode(GL10.GL_MODELVIEW);
		
		gl.glEnable(GL10.GL_TEXTURE_2D);
		
		gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);  
		
		gl.glEnableClientState(GL10.GL_TEXTURE_COORD_ARRAY);

		gl.glVertexPointer(3, GL10.GL_FLOAT, 0, mVertexBuffer);
		
		gl.glTexCoordPointer(2, GL10.GL_FLOAT, 0, mTextureBuffer);
		
		synchronized ( mFrameLock )
		{
			if ( mFrameFlag )
			{
				mFrameFlag = false;
				
				GlColorSpace.toRGB(mTexture, mFrame, mWidth, mHeight);
			}
		}
		
		bindTexture(gl);
		
		gl.glDrawArrays(GL10.GL_TRIANGLE_STRIP, 0, 4);
		
		gl.glDisable(GL10.GL_TEXTURE_2D);
		
		gl.glDisableClientState(GL10.GL_TEXTURE_COORD_ARRAY);
		
		gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
		
		/* FPS METER */
		mNumFrames++;
		
		long fpsElapsed = System.currentTimeMillis() - mFpsStart;
		
		if ( fpsElapsed > 5000 )
		{
			float fps = ( mNumFrames * 1000.0f ) / fpsElapsed;
			
			Log.d("FPS", "fps: " + fps);
			
			mFpsStart = System.currentTimeMillis();
			
			mNumFrames = 0;
		}
	}
	
	private void bindTexture(GL10 gl)
	{
		synchronized ( mTexture )
		{
			if ( mTextureId == null )
			{
				mTextureId = new int[1];
			}
			
			else
			{
				gl.glDeleteTextures(1, mTextureId, 0);
			}
			
			gl.glGenTextures(1, mTextureId, 0);
			
			gl.glBindTexture(GL10.GL_TEXTURE_2D, mTextureId[0]);
			
			switch ( mTextureType )
			{
				case COLOR_ARGB_8888:
					GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, GL10.GL_RGBA, mTexture, GL10.GL_UNSIGNED_BYTE, 0);
					break;
					
				case COLOR_RGB_565:
					GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, GL10.GL_RGB, mTexture, GL10.GL_UNSIGNED_SHORT_5_6_5, 0);
					break;	
	
				case COLOR_GRAY:
					GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, GL10.GL_LUMINANCE, mTexture, GL10.GL_UNSIGNED_BYTE, 0);
					break;
			}
			
			gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_MIN_FILTER, GL10.GL_LINEAR);
		}
	}

	@Override
	public void onSurfaceChanged(GL10 gl, int width, int height)
	{
		gl.glViewport(0, 0, width, height);
		
		gl.glMatrixMode(GL10.GL_PROJECTION);
		
		gl.glLoadIdentity();
		
		gl.glFrustumf(mLeft, mRight, mBottom, mTop, mNear, mFar);
		
		gl.glMatrixMode(GL10.GL_MODELVIEW);
		
		gl.glLoadIdentity();
		
		GLU.gluLookAt(gl, mEye[0], mEye[1], mEye[2], mCenter[0], mCenter[1], mCenter[2], mUp[0], mUp[1], mUp[2]);
	}

	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		gl.glClearColor(1, 1, 1, 1);
		
		gl.glHint(GL10.GL_PERSPECTIVE_CORRECTION_HINT, GL10.GL_FASTEST);
	}
	
	public void build()
	{
		int i = 0;
		
		int maxDim = Math.max(mWidth, mHeight);
		
		mTextureSize = 0;
		
		while ( maxDim > mTextureSize )
		{
			mTextureSize = (int) Math.pow(2, i++);
		}
		
		mVertexCoords = new float[] { -.5f, -.5f, 0,
									   .5f, -.5f, 0, 
									  -.5f,  .5f, 0,
									   .5f,  .5f, 0 };
		
		mTextureCoords = ( !mInvertH ) ? new float[] { 1, 0, 0, 0, 1, 1, 0, 1 } : new float[] { 0, 0, 1, 0, 0, 1, 1, 1 };										
		
		ByteBuffer vb = ByteBuffer.allocateDirect(mVertexCoords.length * 4);
		
		vb.order(ByteOrder.nativeOrder());
		
		mVertexBuffer = vb.asFloatBuffer();
		
		mVertexBuffer.put(mVertexCoords);
		
		mVertexBuffer.position(0);
		
		ByteBuffer tb = ByteBuffer.allocateDirect(mTextureCoords.length * 4);
		
		tb.order(ByteOrder.nativeOrder());
		
		mTextureBuffer = tb.asFloatBuffer();
		
		mTextureBuffer.put(mTextureCoords);
		
		mTextureBuffer.position(0);

		mTextureId = new int[1];
		
		switch ( mTextureType )
		{
			case COLOR_ARGB_8888:
				mTexture = Bitmap.createBitmap(mTextureSize, mTextureSize, Bitmap.Config.ARGB_8888);
				break;
			
			case COLOR_RGB_565:
				mTexture = Bitmap.createBitmap(mTextureSize, mTextureSize, Bitmap.Config.RGB_565);
				break;
				
			case COLOR_GRAY:
				mTexture = Bitmap.createBitmap(mTextureSize, mTextureSize, Bitmap.Config.ALPHA_8);
				break;
		}
	}
	
	public void setColorMode(int pMode)
	{
		synchronized ( mTexture )
		{
			mTextureType = pMode;
			
			build();
		}
	}
	
	public void invertH()
	{
		mInvertH = !mInvertH;
		
		build();
	}
}
