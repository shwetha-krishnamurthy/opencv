#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;
using namespace std;

Mat src, dst, dst1, src_gray;
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Mat abs_grad_x1, abs_grad_y1;
Mat grad;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

void Threshold( int, void* )
{
	/* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( dst , dst1, threshold_value, max_BINARY_value,threshold_type );
  fastNlMeansDenoising(dst1, dst1, 5);

  imshow( "Binary Threshold", dst1 );
}

void ridgeDetect()
{
	int i, j; 
	double gap;

	GaussianBlur( src_gray, src_gray, Size(3,3), 0, 0, BORDER_DEFAULT );

	Mat channels_bgr[3];
	split(src_gray, channels_bgr);

	for(int k = 0; k < 3; k++)
	{
		Sobel( channels_bgr[k], grad_x, ddepth, 2, 0, 3, scale, delta, BORDER_DEFAULT );
	    convertScaleAbs( grad_x, abs_grad_x );

	    Sobel( channels_bgr[k], grad_y, ddepth, 0, 2, 3, scale, delta, BORDER_DEFAULT );
	    convertScaleAbs( grad_y, abs_grad_y );

	    Sobel( channels_bgr[k], grad_x, ddepth, 1, 1, 3, scale, delta, BORDER_DEFAULT );
	    convertScaleAbs( grad_x, abs_grad_x1 );

	    Sobel( channels_bgr[k], grad_y, ddepth, 1, 1, 3, scale, delta, BORDER_DEFAULT );
	    convertScaleAbs( grad_y, abs_grad_y1 );

	    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, channels_bgr[k] );

	    imshow("Sobel", channels_bgr[k]);
	  
	    for (i=1; i < channels_bgr[k].rows; i++)
	    {
	    	for(j=1; j < channels_bgr[k].cols; j++)
	    	{ 
	    		gap = sqrt(abs_grad_x.at<uchar>(i,j)*abs_grad_x.at<uchar>(i,j) 
	    			   + 4*abs_grad_y1.at<uchar>(i,j)*abs_grad_x1.at<uchar>(i,j) 
	    			   - 2*abs_grad_x.at<uchar>(i,j)*abs_grad_y.at<uchar>(i,j) 
	    			   + abs_grad_y.at<uchar>(i,j)*abs_grad_y.at<uchar>(i,j));
	        	dst.at<Vec3b>(i, j).val[k] = 0.5*(abs_grad_x.at<uchar>(i,j) + abs_grad_y.at<uchar>(i,j) + gap);
	    	}
	    }
	}
    convertScaleAbs(dst, dst);
}

int main()
{
  string fname;
  cout << "Enter file name: ";
  cin >> fname;
	src = imread(fname);
  Size dsize=Size(round(0.5*src.rows), round(0.5*src.cols));
  resize(src, src, dsize);
  //src_gray.create( src.size(), src.type() );
  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2HSV);

  // Shadow removal
  Mat channel[3];
	split(src_gray, channel);
	channel[2] = Mat(src_gray.rows, src_gray.cols, CV_8UC1, 200);//Set V

	//Merge channels
	merge(channel, 3, src_gray);
    
  cvtColor (src_gray, src_gray, CV_HSV2BGR);
 //   cvtColor (src_gray, src_gray, CV_BGR2GRAY);
  imshow("Original Image", src_gray);
   
    //Histogram equalization
//    equalizeHist(src_gray, src_gray);
    //merge(channels_bgr, 3, src_gray);
//	imshow( "Histogram Equalization", src_gray );

  dst.create( src_gray.size(), src_gray.type() );

    //Ridge detection function call
  ridgeDetect();
  fastNlMeansDenoising(dst, dst, 5);

  imshow("RidgeDetect", dst);

    /// Create a window to display results
  namedWindow("Binary Threshold", CV_WINDOW_AUTOSIZE );

    /// Create Trackbar to choose type of Threshold
  createTrackbar("Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted",
                  "Binary Threshold", &threshold_type,
                  max_type, Threshold);

  createTrackbar("Value",
                  "Binary Threshold", &threshold_value,
                  max_value, Threshold );

    /// Call the function to initialize
  Threshold( 0, 0 );


  waitKey(0);
}