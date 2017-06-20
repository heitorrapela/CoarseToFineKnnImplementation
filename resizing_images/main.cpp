#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <QDirIterator>
#include <QDebug>

#define WINDOW_NAME "resizing_images"

#define SIZE 20

int main()
{
    QDirIterator it("input", QStringList() << "*.jpg", QDir::Files, QDirIterator::Subdirectories);
    cv::namedWindow(WINDOW_NAME);
    unsigned long int imgCounter = 0;
    while (it.hasNext()) {
        std::cout << "Image counter: " << std::to_string(imgCounter++);
        std::string imgName, outImgName;
        imgName = it.next().toStdString();
        outImgName = "output" + imgName.substr(imgName.find('/'), imgName.rfind(".")-imgName.find('/')) + ".jpg";
        std::cout << "\t" << imgName << " -> " << outImgName;
        cv::Mat image;
        image = cv::imread(imgName);   // Read the file
        if(!image.data)
        {
            std::cout << "\tNo image!" << std::endl;
            image = cv::Mat::zeros(240, 320, CV_8UC3);
            //cv::putText(image, "No image!", cv::Point(80, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255));
            cv::putText(image, "Lena not found!", cv::Point(30, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255));
        }
        else
        {
            std::cout << "\tImage loaded!" << std::endl;
            cv::Mat newImg;
            cv::resize(image, newImg, cv::Size(SIZE, SIZE));
            cv::imwrite(outImgName, newImg);
        }

//        cv::imshow(WINDOW_NAME, image);
//        cv::waitKey(10);
    }

    return 0;
}
