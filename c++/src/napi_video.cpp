#include <napi.h>
#include <unordered_map>
#include <mutex>
#include "video.hpp"
#include <sstream>

class VideoAsyncWorker : public Napi::AsyncWorker {
private:
    Napi::ThreadSafeFunction hitTsfn;
    Napi::ThreadSafeFunction frameTsfn;
    std::string videoFile;
    Options options;
    int devId;

public:
    VideoAsyncWorker(Napi::Function& callback, Napi::Env env, Napi::Function& jsFunction, int devId, std::string videoFile, Options optionsArg, Napi::Function& jsFunctionFrame)
     : Napi::AsyncWorker(callback), devId(devId), videoFile(videoFile), options(optionsArg) {
        hitTsfn = Napi::ThreadSafeFunction::New(env, jsFunction, "jsFunction", 0, 1);
        if (!jsFunctionFrame.IsEmpty()) {
            frameTsfn = Napi::ThreadSafeFunction::New(env, jsFunctionFrame, "jsFunctionFrame", 0, 1);
        }
    }
    virtual ~VideoAsyncWorker(){};

    volatile bool isCancelled = false;
    volatile bool isReset = false;

    void Execute() {
        auto onHit = [this](int hit, float x, float y, int frameNumber, float time) {
            std::string message = "Hit " + std::to_string(hit) + ": " + std::to_string(x) + ", " + std::to_string(y) 
                + " at time " + std::to_string(time) + "\n";
            std::cout << message;
            hitTsfn.BlockingCall([hit, x, y, frameNumber, time](Napi::Env env, Napi::Function jsFunction) {
                Napi::Object obj = Napi::Object::New(env);

                obj.Set("hit", Napi::Number::New(env, hit));
                obj.Set("x", Napi::Number::New(env, x));
                obj.Set("y", Napi::Number::New(env, y));
                obj.Set("time", Napi::Number::New(env, time));

                jsFunction.Call({obj});
            });
        };

        auto onDrawFrame = [this](cv::Mat frame) -> bool {
            if (frameTsfn != nullptr) {
                frameTsfn.BlockingCall([frame](Napi::Env env, Napi::Function jsFunctionFrame) {
                    std::vector<uchar> buf;
                    cv::imencode(".jpg", frame, buf); 

                    Napi::Buffer<uchar> buffer = Napi::Buffer<uchar>::Copy(env, buf.data(), buf.size());

                    Napi::Object obj = Napi::Object::New(env);
                    obj.Set("image", buffer);

                    jsFunctionFrame.Call({obj});
                });
            }
            
            if (isReset)
            {
                // TODO: Inform processVideo to reset using string return (not boolean)
                // isReset = false;
            }
                
            return !isCancelled;
        };

        // std::cout << "worker Threshold Value: " << options.thresholdValue << std::endl;

        if(devId >= 0) {
            processVideo(devId, onHit, onDrawFrame, options);
        }
        else {
            processVideo(videoFile.c_str(), onHit, onDrawFrame, options);
        }
    };

    void OnOK() {
        Napi::HandleScope scope(Env());
        Callback().Call({Env().Null(), Napi::String::New(Env(), "Done")});
    };
};

static std::unordered_map<int, VideoAsyncWorker*> workerMap;
static std::mutex mapMutex;
static int nextWorkerId = 0;

Options ExtractOptions(const Napi::Object& optionsObj) {
    Options options;

    if (optionsObj.Has("thresholdValue")) {
        options.thresholdValue = *(new int(optionsObj.Get("thresholdValue").As<Napi::Number>().Int32Value()));
    }
    if (optionsObj.Has("sharpen")) {
        options.sharpen = *(new int(optionsObj.Get("sharpen").As<Napi::Number>().Int32Value()));
    }
    if (optionsObj.Has("pointLimit")) {
        options.pointLimit = *(new float(optionsObj.Get("pointLimit").As<Napi::Number>().FloatValue()));
    }

    return options;
}

Napi::Value StartAsyncWork(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Function callback = info[0].As<Napi::Function>();
    Napi::Value value = info[1];
    Napi::Function callback2 = info[2].As<Napi::Function>();

    Options options;
    if (info[3].IsObject()) {
        Napi::Object optionsObj = info[3].As<Napi::Object>();
        options = ExtractOptions(optionsObj);
    }

    // std::cout << "startAsyncWork Threshold Value: " << options.thresholdValue << std::endl;

    Napi::Function frameCallback;
    if (info.Length() > 4 && info[4].IsFunction()) {
        frameCallback = info[4].As<Napi::Function>();
    }

    int devId = -1;
    std::string videoFile = "";
    if(value.IsNumber())
        devId = value.As<Napi::Number>().Int32Value();
    else
        videoFile = value.ToString().Utf8Value();

    VideoAsyncWorker* worker = new VideoAsyncWorker(callback, env, callback2, devId, videoFile, options, frameCallback);
    std::lock_guard<std::mutex> guard(mapMutex);
    int workerId = nextWorkerId++;
    workerMap[workerId] = worker;
    worker->Queue();

    return Napi::Number::New(env, workerId);
}

/*Napi::Value UpdateOptions(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int workerId = info[0].As<Napi::Number>().Int32Value();

    Options options;
    if (info[1].IsObject()) {
        Napi::Object optionsObj = info[1].As<Napi::Object>();
        options = ExtractOptions(optionsObj);
    }

    VideoAsyncWorker* worker = workerMap[workerId];

    worker->isReset = true;
    worker->options = options;

    std::lock_guard<std::mutex> guard(mapMutex);
    workerMap.erase(workerId);

    return env.Undefined();
}*/

Napi::Value Close(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int workerId = info[0].As<Napi::Number>().Int32Value();

    VideoAsyncWorker* worker = workerMap[workerId];

    worker->isCancelled = true;

    std::lock_guard<std::mutex> guard(mapMutex);
    workerMap.erase(workerId);

    return env.Undefined();
}

Napi::Value End(const Napi::CallbackInfo& info) {
    return Close(info);
}

Napi::Value ListCameras(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string cameras = listCameras();
    return Napi::String::New(env, cameras);
}

Napi::Value TestOptions(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Options options;

    if (info.Length() > 0 && info[0].IsObject()) {
        Napi::Object optionsObj = info[0].As<Napi::Object>();
        options = ExtractOptions(optionsObj);
    }

    std::stringstream ss;
    ss << "Options: ";
    ss << "ThresholdValue: " << options.thresholdValue << " | ";
    ss << "Sharpen: " << options.sharpen << " | ";
    ss << "Point limit: " << options.pointLimit;

    return Napi::String::New(env, ss.str());
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("processVideo", Napi::Function::New(env, StartAsyncWork));
    // exports.Set("updateOptions", Napi::Function::New(env, UpdateOptions));
    exports.Set("listCameras", Napi::Function::New(env, ListCameras));
    exports.Set("testOptions", Napi::Function::New(env, TestOptions));
    exports.Set("close", Napi::Function::New(env, Close));
    exports.Set("end", Napi::Function::New(env, End));
    return exports;
}

NODE_API_MODULE(video, Init)