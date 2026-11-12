#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    // 1. 加载模型
    string model_path = "../best.onnx";
    Net net = readNetFromONNX(model_path);
    if (net.empty()) {
        cerr << "模型加载失败！" << endl;
        return -1;
    }

    // 2. 准备测试图片列表（确保这些图片在 detect 文件夹里）
    vector<string> image_paths = {
        "../test.jpg", 
        "../test1.jpg", 
        "../test2.jpg", 
        "../test3.jpg"
    };

    string class_names[] = {"blue3", "blue1", "bluesb", "red3", "red1", "redsb"};
    float conf_threshold = 0.5;
    int num_boxes = 8400;
    int dimensions = 10;

    // 3. 循环读取图片（模拟视频流）
    for (size_t img_idx = 0; img_idx < image_paths.size(); ++img_idx) {
        Mat frame = imread(image_paths[img_idx]);
        if (frame.empty()) {
            cout << "跳过无法读取的图片: " << image_paths[img_idx] << endl;
            continue;
        }

        int img_w = frame.cols;
        int img_h = frame.rows;

        // 记录开始时间
        auto start = chrono::high_resolution_clock::now();

        // --- 推理 ---
        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);
        Mat outputs = net.forward();

        // --- 解析 ---
        float* data = (float*)outputs.data;
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> class_ids;

        for (int i = 0; i < num_boxes; ++i) {
            float max_class_score = 0;
            int max_class_id = -1;
            for (int c = 0; c < 6; ++c) {
                float score = data[(4 + c) * num_boxes + i];
                if (score > max_class_score) {
                    max_class_score = score;
                    max_class_id = c;
                }
            }

            if (max_class_score > conf_threshold) {
                float cx = data[0 * num_boxes + i];
                float cy = data[1 * num_boxes + i];
                float w  = data[2 * num_boxes + i];
                float h  = data[3 * num_boxes + i];

                float scale_x = (float)img_w / 640.0;
                float scale_y = (float)img_h / 640.0;
                cx *= scale_x; cy *= scale_y; w *= scale_x; h *= scale_y;

                int left = int(cx - w / 2.0);
                int top = int(cy - h / 2.0);

                boxes.push_back(Rect(left, top, int(w), int(h)));
                confidences.push_back(max_class_score);
                class_ids.push_back(max_class_id);
            }
        }

        // --- NMS ---
        vector<int> indices;
        NMSBoxes(boxes, confidences, conf_threshold, 0.4, indices);

        // --- 绘制 ---
        for (int idx : indices) {
            Rect box = boxes[idx];
            int class_id = class_ids[idx];
            
            int center_x = box.x + box.width / 2;
            int center_y = box.y + box.height / 2;
            Point center(center_x, center_y);

            rectangle(frame, box, Scalar(0, 255, 0), 2);
            circle(frame, center, 4, Scalar(0, 0, 255), -1); // 红色中心点
            
            string label = class_names[class_id] + " " + to_string(int(confidences[idx] * 100)) + "%";
            putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        // --- 计算 FPS ---
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;
        double fps = 1000.0 / elapsed.count();
        
        string fps_text = "FPS: " + to_string(int(fps));
        putText(frame, fps_text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

        // --- 显示 ---
        imshow("RM Vision - Image " + to_string(img_idx + 1), frame);
        
        // 停留 1 秒钟，方便你看清楚
        waitKey(1000); 
    }

    destroyAllWindows();
    cout << "所有图片测试完成！" << endl;
    return 0;
}