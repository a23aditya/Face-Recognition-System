all:
	g++ face_recognition_main.cpp -o face_app `pkg-config --cflags --libs opencv4` -std=c++11

