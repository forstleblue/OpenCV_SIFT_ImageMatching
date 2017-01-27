
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// Load the two images.
	// 'query' and 'train' are the notation used by the parameters in the 'match' function.
	// It seems backwards from how I'm applying it--the first image is where I've
	// isolated the object I'm looking for, and the second is the image I want to locate that
	// object in.
	Mat queryImg = imread("..\\Images\\frame_18_penguin.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat trainImg = imread("..\\Images\\frame_20.png", CV_LOAD_IMAGE_GRAYSCALE);
    
	// Verify the images loaded successfully.
    if(queryImg.empty() || trainImg.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    // Detect keypoints in both images.
    SiftFeatureDetector detector(400);
    vector<KeyPoint> queryKeypoints, trainKeypoints;
    detector.detect(queryImg, queryKeypoints);
    detector.detect(trainImg, trainKeypoints);

	// Print how many keypoints were found in each image.
	printf("Found %d and %d keypoints.\n", queryKeypoints.size(), trainKeypoints.size());

    // Compute the SIFT feature descriptors for the keypoints.
	// Multiple features can be extracted from a single keypoint, so the result is a
	// matrix where row 'i' is the list of features for keypoint 'i'.
    SiftDescriptorExtractor extractor;
    Mat queryDescriptors, trainDescriptors;
    extractor.compute(queryImg, queryKeypoints, queryDescriptors);
    extractor.compute(trainImg, trainKeypoints, trainDescriptors);

	// Print some statistics on the matrices returned.
	Size size = queryDescriptors.size();
	printf("Query descriptors height: %d, width: %d, area: %d, non-zero: %d\n", 
		   size.height, size.width, size.area(), countNonZero(queryDescriptors));
	
	size = trainDescriptors.size();
	printf("Train descriptors height: %d, width: %d, area: %d, non-zero: %d\n", 
		   size.height, size.width, size.area(), countNonZero(trainDescriptors));

    // For each of the descriptors in 'queryDescriptors', find the closest 
	// matching descriptor in 'trainDescriptors' (performs an exhaustive search).
	// This seems to only return as many matches as there are keypoints. For each
	// keypoint in 'query', it must return the descriptor which most closesly matches a
	// a descriptor in 'train'?
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(queryDescriptors, trainDescriptors, matches);

	printf("Found %d matches.\n", matches.size());

    // Draw the results. Displays the images side by side, with colored circles at
	// each keypoint, and lines connecting the matching keypoints between the two 
	// images.
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(queryImg, queryKeypoints, trainImg, trainKeypoints, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}
