#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

/* ---------------- Configuration ---------------- */
#define DB_PATH "./face_db"
#define CASCADE_FILE "haarcascade_frontalface_default.xml"
#define IMG_WIDTH  200
#define IMG_HEIGHT 200
/* ------------------------------------------------ */

CascadeClassifier face_cascade;
Ptr<LBPHFaceRecognizer> model;

/* -------- Utility: create directory if not exist -------- */
void ensure_dir(const string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        mkdir(path.c_str(), 0755);
}

/* -------- Capture faces for a user -------- */
void register_face(int user_id)
{
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Camera open failed\n";
        return;
    }

    string user_dir = string(DB_PATH) + "/" + to_string(user_id);
    ensure_dir(DB_PATH);
    ensure_dir(user_dir);

    cout << "Registering face for user " << user_id << endl;
    cout << "Press 'c' to capture, 'q' to quit\n";

    Mat frame, gray;
    int count = 0;

    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (auto &f : faces) {
            rectangle(frame, f, Scalar(0,255,0), 2);

            Mat face = gray(f);
            resize(face, face, Size(IMG_WIDTH, IMG_HEIGHT));

            if (waitKey(1) == 'c') {
                string filename = user_dir + "/" + to_string(count++) + ".png";
                imwrite(filename, face);
                cout << "Saved: " << filename << endl;
            }
        }

        imshow("Register", frame);
        if (waitKey(1) == 'q') break;
    }

    destroyAllWindows();
}

/* -------- Load images and train model -------- */
void train_model()
{
    vector<Mat> images;
    vector<int> labels;

    DIR *db = opendir(DB_PATH);
    if (!db) {
        cerr << "No face database found\n";
        return;
    }

    struct dirent *user;
    while ((user = readdir(db)) != NULL) {
        if (user->d_type != DT_DIR) continue;
        if (user->d_name[0] == '.') continue;

        int label = atoi(user->d_name);
        string user_path = string(DB_PATH) + "/" + user->d_name;

        DIR *ud = opendir(user_path.c_str());
        struct dirent *img;

        while ((img = readdir(ud)) != NULL) {
            if (img->d_name[0] == '.') continue;

            string img_path = user_path + "/" + img->d_name;
            Mat im = imread(img_path, IMREAD_GRAYSCALE);
            if (im.empty()) continue;

            images.push_back(im);
            labels.push_back(label);
        }
        closedir(ud);
    }
    closedir(db);

    if (images.empty()) {
        cerr << "No training data\n";
        return;
    }

    model = LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->save("face_model.yml");

    cout << "Training completed\n";
}

/* -------- Recognize face -------- */
void recognize_face()
{
    model = LBPHFaceRecognizer::create();
    model->read("face_model.yml");

    VideoCapture cam(0);
    if (!cam.isOpened()) {
        cerr << "Camera open failed\n";
        return;
    }

    Mat frame, gray;

    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (auto &f : faces) {
            Mat face = gray(f);
            resize(face, face, Size(IMG_WIDTH, IMG_HEIGHT));

            int label;
            double confidence;
            model->predict(face, label, confidence);

            rectangle(frame, f, Scalar(255,0,0), 2);
            putText(frame,
                    "ID: " + to_string(label) + " conf: " + to_string(confidence),
                    Point(f.x, f.y - 10),
                    FONT_HERSHEY_SIMPLEX,
                    0.6, Scalar(0,255,0), 2);
        }

        imshow("Recognize", frame);
        if (waitKey(1) == 'q') break;
    }

    destroyAllWindows();
}

/* ------------------- MAIN ------------------- */
int main()
{
    if (!face_cascade.load(CASCADE_FILE)) {
        cerr << "Failed to load Haar cascade\n";
        return -1;
    }

    while (1) {
        cout << "\n1. Register Face\n";
        cout << "2. Train Model\n";
        cout << "3. Recognize Face\n";
        cout << "0. Exit\n";
        cout << "Choice: ";

        int ch;
        cin >> ch;

        if (ch == 1) {
            int uid;
            cout << "Enter User ID: ";
            cin >> uid;
            register_face(uid);
        }
        else if (ch == 2) {
            train_model();
        }
        else if (ch == 3) {
            recognize_face();
        }
        else if (ch == 0) {
            break;
        }
    }
    return 0;
}

