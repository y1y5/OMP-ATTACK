#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

extern "C"
{

    const float x_MIN = 0.0;
    const float x_MAX = 70.0;
    const float y_MIN = -40.0;
    const float y_MAX = 40.0;
    const float z_MIN = -2.5;
    const float z_MAX = 1;
    const float x_DIVISION = 0.1;
    const float y_DIVISION = 0.1;
    const float z_DIVISION = 0.1;

    int X_SIZE = (int)((x_MAX - x_MIN) / x_DIVISION);
    int Y_SIZE = (int)((y_MAX - y_MIN) / y_DIVISION);
    int Z_SIZE = (int)((z_MAX - z_MIN) / z_DIVISION);

    inline int getX(float x) {
        return (int)((x - x_MIN) / x_DIVISION);
    }

    inline int getY(float y) {
        return (int)((y - y_MIN) / y_DIVISION);
    }

    inline int getZ(float z) {
        return (int)((z - z_MIN) / z_DIVISION);
    }

    inline int at3(int a, int b, int c) {
        return a * (X_SIZE * (Z_SIZE + 1)) + b * (Z_SIZE + 1) + c;
    }

    inline int at2(int a, int b) {
        return a * X_SIZE + b;
    }

    void createTopViewMaps(const void* indatav, const char* path, const void* add, int add_len) {
        float* data_cube = (float*) indatav;
        float* data = nullptr;
        FILE* fp = fopen(path, "rb");

        if(fp == NULL) {
            cout << path << " not found. Ensure that the file path is correct." << endl;
            return;
        }

        int32_t num = 1000000;
        data = (float*)malloc(num * sizeof(float));
        num = fread(data, sizeof(float), num, fp) / 4;
        fclose(fp);

        float* px = data + 0;
        float* py = data + 1;
        float* pz = data + 2;
        float* pr = data + 3;

        // Height features X_SIZE * Y_SIZE * (Z_SIZE + 1)
        // Density feature X_SIZE * Y_SIZE * 1
        std::vector<int> density_map(Y_SIZE * X_SIZE);

        for (int32_t i = 0; i < num; ++i) {
            if (*px > x_MIN && *py > y_MIN && *pz > z_MIN && *px < x_MAX && *py < y_MAX && *pz < z_MAX) {
                int X = getX(*px);
                int Y = getY(*py);
                int Z = getZ(*pz);
                *(data_cube + at3(Y, X, Z)) = 1;
                *(data_cube + at3(Y, X, Z_SIZE)) += *pr;
                density_map[at2(Y, X)]++;
            }
            px += 4; py += 4; pz += 4; pr += 4;
        }

        float* add_data = (float*) add;
        for (int32_t i = 0; i < add_len; ++i) {
            float px = add_data[i * 4 + 0];
            float py = add_data[i * 4 + 1];
            float pz = add_data[i * 4 + 2];
            float pr = add_data[i * 4 + 3];

            if (px > x_MIN && py > y_MIN && pz > z_MIN && px < x_MAX && py < y_MAX && pz < z_MAX) {
                int X = getX(px);
                int Y = getY(py);
                int Z = getZ(pz);
                *(data_cube + at3(Y, X, Z)) = 1;
                *(data_cube + at3(Y, X, Z_SIZE)) += pr;
                density_map[at2(Y, X)]++;
            }
        }

        for (int y = 0; y < Y_SIZE; ++y) {
            for (int x = 0; x < X_SIZE; ++x) {
                if (density_map[at2(y, x)] > 0) {
                    *(data_cube + at3(y, x, Z_SIZE)) /= (float)density_map[at2(y, x)];
                }
            }
        }

        free(data);
    }
}
