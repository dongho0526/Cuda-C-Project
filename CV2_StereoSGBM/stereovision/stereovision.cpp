//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//    // 이미지 로드
//    string left_image_path = "000000_10_L.png";
//    string right_image_path = "000000_10_R.png";
//
//    Mat left_img = imread(left_image_path, IMREAD_GRAYSCALE);
//    Mat right_img = imread(right_image_path, IMREAD_GRAYSCALE);
//
//    if (left_img.empty() || right_img.empty()) {
//        cerr << "Could not open or find the images!" << endl;
//        return -1;
//    }
//
//    // SGBM 파라미터 설정
//    int min_disp = 0;
//    int num_disp = 16 * 6;  // 16의 배수여야 함
//    int block_size = 5;
//    int P1 = 8 * 3 * block_size * block_size;
//    int P2 = 32 * 3 * block_size * block_size;
//    int disp12MaxDiff = 1;
//    int uniquenessRatio = 10;
//    int speckleWindowSize = 100;
//    int speckleRange = 32;
//
//    // StereoSGBM 객체 생성
//    Ptr<StereoSGBM> stereo = StereoSGBM::create(min_disp, num_disp, block_size);
//    stereo->setP1(P1);
//    stereo->setP2(P2);
//    stereo->setDisp12MaxDiff(disp12MaxDiff);
//    stereo->setUniquenessRatio(uniquenessRatio);
//    stereo->setSpeckleWindowSize(speckleWindowSize);
//    stereo->setSpeckleRange(speckleRange);
//
//    // 디스패리티 맵 계산
//    Mat disparity;
//    stereo->compute(left_img, right_img, disparity);
//
//    // 디스패리티 맵 정규화
//    Mat disparity_normalized;
//    normalize(disparity, disparity_normalized, 0, 255, NORM_MINMAX, CV_8U);
//
//    // 디스패리티 맵 색상 변환
//    Mat color_disparity_map;
//    applyColorMap(disparity_normalized, color_disparity_map, COLORMAP_JET);
//
//    // 결과 시각화
//    imshow("Left Image", left_img);
//    imshow("Disparity Map (SGBM)", color_disparity_map);
//    waitKey(0);
//
//    return 0;
//}

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the two images
    cv::Mat imgL = cv::imread("im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread("im1.png", cv::IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Error reading images!" << std::endl;
        return -1;
    }

    // Parameters for SGBM
    int numDisparities = 16 * 5;  // must be divisible by 16
    int blockSize = 11;

    // Create the SGBM object
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);
    sgbm->setP1(8 * imgL.channels() * blockSize * blockSize);
    sgbm->setP2(32 * imgL.channels() * blockSize * blockSize);
    sgbm->setPreFilterCap(63);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

    // Time measurement start
    cv::TickMeter tm;
    tm.start();

    // Compute the disparity map
    cv::Mat disparity;
    sgbm->compute(imgL, imgR, disparity);

    // Time measurement stop
    tm.stop();
    std::cout << "StereoVision Depth Map(OpenCV) Time for Excution: " << tm.getTimeMilli() << " ms" << std::endl;

    // Normalize the disparity map for visualization
    cv::Mat disp8;
    disparity.convertTo(disp8, CV_8U, 255 / (numDisparities * 16.));

    // Apply a color map to the disparity map
    cv::Mat dispColor;
    applyColorMap(disp8, dispColor, cv::COLORMAP_JET);

    // Display the images and disparity map
    cv::imshow("Left Image", imgL);
    cv::imshow("Right Image", imgR);
    cv::imshow("Disparity Map", dispColor);

    // Save the disparity map
    cv::imwrite("output_img.png", dispColor);

    cv::waitKey(0);
    return 0;
}