
#include<stdio.h>


void removeNoise(Mat src, int var) {

	imshow("original", src);
	Mat src2 = src.clone();
	int mean = 0;
	double noise = 0;
	double noisePer = 0;

	for (int i = 0 ; i < src.rows; i++) {
		for (int j = 0 ; j < src.cols; j++) {

			// 3x3 kernel
			int c = 0;
			for (int x = i-1 ; x <= i+1 ; x++) {
				for (int y = j-1 ; y <= j+1 ; y++) {
					if(x>0 && y>0 && x<src.rows && y<src.cols){
						mean += src.at<uchar>(x, y);
						c++;
					}
				}
			}

			mean = mean / c;
			int variance = (src.at<uchar>(i, j) - mean) * (src.at<uchar>(i, j) - mean);
			if (variance > var) {
				//src2.at<uchar>(i, j) = 0;
				noise++;
			}
		}
	}

	noisePer = (noise / 2304000) * 100;

	// Detecting noise and correcting it
	if (noisePer > 4) {
		cout << "Noise percentage: " << noisePer << endl;
		medianBlur(src, src, 5);
	}
	else cout << "Input image has no noise" << endl;
}

void removeBlur(Mat src) {

	imshow("Original", src);
	Mat sobel, sobelX, sobelY;
	Mat abs_SobelX, abs_SobelY;

	Sobel(src, sobelX, CV_32F, 1, 0);
	convertScaleAbs(sobelX, abs_SobelX);

	Sobel(src, sobelY, CV_32F, 0, 1);
	convertScaleAbs(sobelY, abs_SobelY);

	addWeighted(abs_SobelX, 0.5, abs_SobelY, 0.5, 0, sobel);


	double white = 0;
	double blurPer = 0;

	for (int i = 0; i < sobel.rows; i++) {
		for (int j = 0; j < sobel.cols; j++) {
			if (sobel.at<uchar>(i, j) > 150)
				white++;
		}
	}

	blurPer = (white / 2304000) * 100;

	// Detecting blur and correcting it
	if (blurPer < 5) {
		cout << "Number of edge pixels = " << white << endl;
		cout << "Percentage of blur in the image = " << blurPer << endl;

		/*
		Mat edges;
		//cv::Mat sharpened(cv::Size(1920,1200), CV_32F);
		Mat sharpened = src.clone();

		//Canny(src,edges,120,20,3);

		Laplacian(src, edges, CV_32F, 3, 3, 0, BORDER_DEFAULT);
		convertScaleAbs(edges, edges);

		imshow("edges", edges);

		for (int i = 0; i < edges.rows; i++) {
			for (int j = 0; j < edges.cols; j++) {
				if (edges.at<uchar>(i, j) > 200) {
					if (src.at<uchar>(i, j) + edges.at<uchar>(i, j) < 255)
						sharpened.at<uchar>(i, j) = src.at<uchar>(i, j) + edges.at<uchar>(i, j);
					else sharpened.at<uchar>(i, j) = (src.at<uchar>(i, j) + edges.at<uchar>(i, j)) - 255;
				}
			}
		}
		*/

		Mat gb;
		GaussianBlur(src,gb,cv::Size(99,99),0);
		addWeighted(src,1.5,gb,-0.5,0,gb);

		imshow("Sharpened", gb);
	}
	else cout << "Input image has no blur" << endl;
}

void removeColorCorrection(Mat src) {

	imshow("Input image", src);
	int histogram[256];
	int cumHistogram[256];

	// Creating an intensity histogram for the image
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			histogram[src.at<uchar>(i, j)]++;
		}
	}


	// Creating a cumulative histogram for the image
	for (int i = 1; i < (sizeof(cumHistogram) / sizeof(*cumHistogram)) - 1; i++) {
		cumHistogram[i] += cumHistogram[i - 1];
	}

	// Searching for the min & max intensity
	int minFreq, maxFreq, minRGB, maxRGB;
	double cRange, cRangePer;
	minFreq = histogram[0];
	maxFreq = histogram[0];
	minRGB = 0;
	maxRGB = 0;
	for (int i=0; i<(sizeof(cumHistogram)/sizeof(*cumHistogram))-1;i++){
		if (histogram[i] > maxFreq) {
			maxFreq = histogram[i];
			maxRGB = i;
		}
		else if (histogram[i] < minFreq) {
			minFreq = histogram[i];
			minRGB = i;
		}
	}

	cout << "Min = " << minRGB << " Max = " << maxRGB << endl;
	cRange = (double) (maxRGB - minRGB);
	cRangePer = (double)(cRange / 255.0)*100.0;

	// Your logic is fucked up, run on image 1 to remember why!

	// Detecting color collapsing and correcting it
	if (cRangePer < 50) {
		cout << "Percentage of colors used: " << cRangePer << endl;
	}
	else cout << "Input image has no color collapsing" << endl;
}
