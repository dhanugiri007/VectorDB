#include <vector>
#include <string>
#include <cmath>

#include <algorithm> 
#include <iostream>

using namespace std;

struct VectorItem {
    int id;
    string label;
    vector<float> values;
};

float cosineDistance(const vector<float> &a, const vector<float> &b) {
    float dot = 0;
    for(int i = 0; i < (int)a.size(); i++) {
        dot += a[i] * b[i];
    }

    float magA = 0;
    float magB = 0;
    for(int i = 0; i < (int)a.size(); i++) {
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    magA = sqrt(magA);
    magB = sqrt(magB);

    if (magA == 0 || magB == 0) return 1.0f; 

    float similarity = dot / (magA * magB);
    return 1.0f - similarity;
}

float euclideanDistance(const vector<float> &a, const vector<float> &b) {
    float sum = 0;
    for(int i = 0; i < (int)a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

float manhattanDistance(const vector<float> &a, const vector<float> &b) {
    float sum = 0;
    for(int i = 0; i < (int)a.size(); i++) {
        sum += abs(a[i] - b[i]);
    }
    return sum;
}

class BruteForce {
public:
    vector<VectorItem> items;

    void insert(const VectorItem& item) {
        items.push_back(item);
    }

    bool remove(int id) {
        auto initial_size = items.size();
        items.erase(
            remove_if(items.begin(), items.end(), 
                [id](const VectorItem& item) { return item.id == id; }), 
            items.end()
        );
        return items.size() < initial_size;
    }

    vector<pair<VectorItem, float>> search(
        const vector<float>& query,
        int k,
        float (*distFunc)(const vector<float>&, const vector<float>&)
    ) {
        vector<pair<VectorItem, float>> results;
        for(const auto& item: items) {
            float dist = distFunc(query, item.values);
            results.push_back({item, dist});
        }

        sort(results.begin(), results.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });

        if((int)results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    size_t size() const {
        return items.size();
    }
};

struct KDNode {
    VectorItem item;
    KDNode* left = nullptr;
    KDNode* right = nullptr;
};

class KDTree {
public: 
    ~KDTree() {
        destroy(root_);
    }

    void build(vector<VectorItem> items) {
        root_ = buildRecursive(items, 0);
    }

    KDNode* getRoot() {
        return root_;
    }

    // Public entry point for search
    vector<pair<VectorItem, float>> search(const vector<float>& query, int k) {
        vector<pair<VectorItem, float>> best;
        searchRecursive(root_, query, 0, k, best);
        return best;
    }

private:
    KDNode* root_ = nullptr;

    KDNode* buildRecursive(vector<VectorItem>& items, int depth) {
        if(items.empty()) return nullptr;

        int dim = depth % items[0].values.size();

        sort(items.begin(), items.end(), [dim](const VectorItem& a, const VectorItem& b) {
            return a.values[dim] < b.values[dim];
        });

        size_t medianIdx = items.size() / 2;

        KDNode* node = new KDNode();
        node->item = items[medianIdx];

        vector<VectorItem> leftItems(items.begin(), items.begin() + medianIdx);
        vector<VectorItem> rightItems(items.begin() + medianIdx + 1, items.end());

        node->left = buildRecursive(leftItems, depth + 1);
        node->right = buildRecursive(rightItems, depth + 1);

        return node;
    }

    void destroy(KDNode* node) {
        if(node == nullptr) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
    }

    void searchRecursive(KDNode* node, const vector<float>& query, int depth,
                         int k, vector<pair<VectorItem, float>>& best) {
        if (node == nullptr) return;

        int dim = depth % query.size();

        KDNode* nearSide;
        KDNode* farSide;
        if (query[dim] < node->item.values[dim]) {
            nearSide = node->left;
            farSide = node->right;
        } else {
            nearSide = node->right;
            farSide = node->left;
        }

        searchRecursive(nearSide, query, depth + 1, k, best);

        float dist = euclideanDistance(query, node->item.values);
        best.push_back({node->item, dist});
        sort(best.begin(), best.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; });
        if ((int)best.size() > k) best.resize(k);

        float axisDist = abs(query[dim] - node->item.values[dim]);
        if ((int)best.size() < k || axisDist < best.back().second) {
            searchRecursive(farSide, query, depth + 1, k, best);
        }
    }
};

void printTree(KDNode* node, int depth = 0) {
    if (node == nullptr) return;
    printTree(node->left, depth + 1);
    cout << string(depth * 2, ' ') << node->item.label << "\n";
    printTree(node->right, depth + 1);
}



int main() {
    vector<float> v1 = {1.0f, 0.0f, 0.0f};
    vector<float> v2 = {0.0f, 1.0f, 0.0f};
    vector<float> v3 = {1.0f, 0.1f, 0.0f};

    cout << "cosine(v1,v2): " << cosineDistance(v1, v2) << endl;
    cout << "cosine(v1,v3): " << cosineDistance(v1, v3) << endl;
    cout << "euclidean(v1,v2): " << euclideanDistance(v1, v2) << endl;
    cout << "manhattan(v1,v2): " << manhattanDistance(v1, v2) << endl;

    BruteForce db;
    db.insert({1, "point A", {1.0f, 0.0f, 0.0f}});
    db.insert({2, "point B", {0.0f, 1.0f, 0.0f}});
    db.insert({3, "point C", {0.9f, 0.1f, 0.0f}});
    db.insert({4, "point D", {0.0f, 0.0f, 1.0f}});
    db.insert({5, "point E", {0.8f, 0.2f, 0.0f}});

    vector<float> query = {1.0f, 0.0f, 0.0f};
    auto results = db.search(query, 3, cosineDistance);
    cout << "\nTop 3 nearest to (1,0,0) by cosine distance:\n";
    for (const auto& pair : results) {
        cout << "  id=" << pair.first.id 
             << " label=" << pair.first.label
             << " dist=" << pair.second << "\n";
    }

    KDTree tree;
    tree.build(db.items);

    auto bfResults = db.search(query, 3, euclideanDistance);
    auto kdResults = tree.search(query, 3);

    cout << "\nBruteForce (euclidean):\n";
    for (auto& pair : bfResults)
        cout << "  " << pair.first.label << " dist=" << pair.second << "\n";

    cout << "\nKDTree (euclidean):\n";
    for (auto& pair : kdResults)
        cout << "  " << pair.first.label << " dist=" << pair.second << "\n";

    return 0;
}