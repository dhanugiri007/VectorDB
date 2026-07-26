#include <vector>
#include <string>
#include <cmath>
#include <algorithm> 
#include <iostream>

struct VectorItem {
    int id;
    std::string label;
    std::vector<float>values;
};

float cosineDistance(const std::vector<float> &a, const std::vector<float> &b) {

    // 1.Dot product of a and b
    float dot = 0;
    for(int i=0; i< (int) a.size(); i++) {
        dot += a[i] * b[i];
    }

    // 2. magnitude  of a, magnitude of b
    float magA = 0;
    float magB = 0;
    for(int i=0; i<(int) a.size(); i++) {
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    magA = sqrt(magA);
    magB = sqrt(magB);

    if (magA == 0 || magB == 0) return 1.0f; 

    //3. cosine simlarity = dot / magA * magB;
    float simlarity = dot / (magA * magB);

    return 1-simlarity;

}
float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0;
    for(int i=0; i<(int) a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}
float manhattanDistance(const std:: vector<float> &a, const std::vector<float> &b) {
    float sum = 0;
    for(int i=0; i<(int) a.size(); i++) {
        sum += abs(a[i]-b[i]);
    }
    return sum;

}



int main() {

    std::vector<float> v1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> v3 = {1.0f, 0.1f, 0.0f};

    std::cout << "cosine(v1,v2): " << cosineDistance(v1, v2);
    std::cout << "cosine(v1,v3): " << cosineDistance(v1, v3);
    std::cout << "euclidean(v1,v2): " << euclideanDistance(v1, v2);
    std::cout << "manhattan(v1,v2): " << manhattanDistance(v1, v2);

    return 0;
}