#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main()
{
	string fname;
	cout << "Enter file name: ";
	cin >> fname;

	// Read image
	Mat im = imread( fname, IMREAD_GRAYSCALE );
	Size dsize=Size(round(0.25*im.rows), round(0.25*im.cols));
    resize(im, im, dsize);

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 256;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 150;

	/*
	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;
	
	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.01;
	/* Y
	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.001;
	*/


	// Storage for blobs
	vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// Detect blobs
	detector.detect( im, keypoints);
#else 

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

	// Detect blobs
	detector->detect( im, keypoints);
#endif 

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;

	drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	cout<<"No. of blobs: "<< keypoints.size()<<"\n";
	imwrite( "blobs.jpg", im_with_keypoints);

	// Show blobs
	imshow("original", im);
	imshow("keypoints", im_with_keypoints);
	waitKey();
}
