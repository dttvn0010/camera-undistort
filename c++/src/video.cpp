#include "video.hpp"

void printHelp(const char* name) {
    std::cout << "Usage: " << name << " [videoFile] [-c videoDevice] [-n]\n\t-c id:\tUse video device id\n\t-n:\tPlay video at normal speed\n";
}

int videoDevice = -1;
char* filename = NULL;
bool normalSpeedFlag = false;

bool handleArguments(int argc, char* argv[]) {
    char* endptr;

    for(int i = 1; i < argc; i++) {
        if(std::strncmp(argv[i], "-c", 2) == 0) {
            if(std::strlen(argv[i]) > 2) {
                videoDevice = strtol(&argv[i][2], &endptr, 10);
                if(*endptr != '\0')
                    goto help;
            }
            else if(i + 1 < argc) {
                i++;
                videoDevice = strtol(argv[i], &endptr, 10);
                if(*endptr != '\0')
                    goto help;
            }
            else {
                goto help;
            }
        }
        else if(std::strcmp(argv[i], "-n") == 0) {
            normalSpeedFlag = true;
        }
        else if(std::strncmp(argv[i], "-", 1) != 0) {
            filename = argv[i];
        }
        else {
            goto help;
        }
    }

    if(!filename  && videoDevice !=  -1)
        goto help;

    return false;

help:
    printHelp(argv[0]);
    return true;
}

bool firstCheck = true;

bool onDrawFrame(cv::Mat frame) {
    const char* title = "Processed Frame";
    //if(cv::getWindowProperty(title, cv::WND_PROP_AUTOSIZE) == -1 && !firstCheck)
    //    return true;
    firstCheck = false;

    cv::imshow(title, frame);
    return(!(cv::waitKey(30) >= 0));
}

void onHitCallback(int hit, float x, float y, int frameNumber, float time) {
    std::cout << "Hit " << hit << ": " << x << ", " << y << ", frame number:" << frameNumber <<  ", at time " << time << "\n";
}

int main(int argc, char* argv[]) {
    if(handleArguments(argc, argv))
        return 1;

    if(videoDevice >= 0)
        processVideo(videoDevice, onHitCallback, nullptr);
    else
        processVideo(filename, onHitCallback, nullptr);

    cv::waitKey();

    cv::destroyAllWindows();
    return 0;
}
