#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

Mat adaptive_threshold(Mat image)
{
    Mat result(image.size(),image.type());
    for(int i = 0 ; i < image.rows ; i++)
    {
        for(int j = 0; j < image.cols ; j++)
        {
            int sum = 0 , avg = 0, t = 9;
            for(int m=i-1; m<=i+1;m++)
            {
                for(int n=j-1;n<=j+1;n++)
                {
                    if(m < 0 || n < 0 || m > image.rows - 1 || n > image.cols - 1)
                    {
                        t--;
                        continue;
                    }
                    sum += image.at<uchar>(m,n);
                }
            }
            avg = sum / t;
            if(image.at<uchar>(i,j)>avg)
                result.at<uchar>(i,j)= 255;
            else
                result.at<uchar>(i,j)= 0;
        }
    }
    return result;
}

void detect_shape(Mat img)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    vector<Point> approx;
    int idx = 0;
    Mat dst(img.size(), CV_8UC3);
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( dst, contours, idx, color, CV_FILLED, 8, hierarchy );
        approxPolyDP(contours[idx], approx, 0.1*arcLength(contours[idx], 1), 1);
        cout << "Number of sides: " << approx.size() <<"\n";
    }

    namedWindow("Contours", WINDOW_AUTOSIZE);
    imshow("Contours",dst);
}

int main()
{
    Mat image, binary;
    string fname;
    //cout << "CV_MAJOR_VERSION: "<< CV_MAJOR_VERSION << "\nCV_MINOR_VERSION: "<< CV_MINOR_VERSION;
    cout << "\nEnter name of file:";
    cin >> fname;
    image = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
    Size dsize=Size(round(0.25*image.rows), round(0.25*image.cols));
    resize(image, image, dsize);
    binary = adaptive_threshold(image);
    namedWindow("Original",WINDOW_AUTOSIZE);
    imshow("Original",image);
    namedWindow("Binary",WINDOW_AUTOSIZE);
    imshow("Binary",binary);
    detect_shape(binary);
    waitKey(0);
    return 0;
}