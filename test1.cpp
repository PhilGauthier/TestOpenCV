#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
 
using namespace std;
using namespace cv;
 
Mat src, src_gray;
Mat dst, detected_edges;


//Canny
int edgeThresh = 1;
int lowThreshold;
int value2;
int const max_lowThreshold = 255;
int const max_ratio = 50;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/// Morpho variables
Mat srcM, dstM, dstBin;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;

char* windowMorpho_name = "Morphology Transformations Demo";

//box variables
RNG rng(12345);

void Morphology_Operations( int, void* )
{
   
    // Since MORPH_X : 2,3,4,5 and 6
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( dst, dstM, operation, element );
    /// Convert the image to grayscale
    /// Detect edges using Threshold
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using Threshold
    
    cvtColor(dstM, threshold_output, CV_RGB2GRAY);
    /// Find contours
    findContours( threshold_output, contours,  CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    { 
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }


    /// Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }

    imshow( windowMorpho_name, drawing );
}





/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
  Morphology_Operations( 0, 0 );
 }

 void ratioUpdate(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
 }



/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  //src = imread( argv[1] );
     Mat imgIn = imread("/home/phil/Pictures/contrastDePhase-stack_axon.jpg",1);
    //equalizeHist(imgIn,imgIn);   

    cv::resize(imgIn, src, Size(640,480));

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  createTrackbar( "Ratio:", window_name, &ratio, max_ratio, ratioUpdate );

  /// Show the image
  CannyThreshold(0, 0);



  /// Create window
 namedWindow( windowMorpho_name, CV_WINDOW_AUTOSIZE );

 /// Create Trackbar to select Morphology operation
 createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", windowMorpho_name, &morph_operator, max_operator, Morphology_Operations );

 /// Create Trackbar to select kernel type
 createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", windowMorpho_name,
                 &morph_elem, max_elem,
                 Morphology_Operations );

 /// Create Trackbar to choose kernel size
 createTrackbar( "Kernel size:\n 2n +1", windowMorpho_name,
                 &morph_size, max_kernel_size,
                 Morphology_Operations );

  /// Default start
  Morphology_Operations( 0, 0 );


  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }