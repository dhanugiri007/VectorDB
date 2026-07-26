#include <vector>
#include <string>
#include <cmath>
#include <algorithm> 
#include <iostream>
#include <queue>
#include <unordered_set>
#include <random>

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

class HNSW {
public:
    HNSW(int M = 8, int efConstruction = 100)
        : M_(M), efConstruction_(efConstruction) {}

    void insert(const VectorItem& item) {
        int newIdx = (int)nodes_.size();
        int l = randomLayer();

        Node newNode;
        newNode.item = item;
        newNode.topLayer = l;
        newNode.neighbors.resize(l + 1);
        nodes_.push_back(newNode);

        if (entryPoint_ == -1) {
            entryPoint_ = newIdx;
            maxLayer_ = l;
            return;
        }

        int currObj = entryPoint_;
        int topL = maxLayer_;

        for (int layer = topL; layer > l; --layer) {
            currObj = greedyClosest(currObj, item.values, layer);
        }

        vector<int> entryPoints = {currObj};
        for (int layer = min(topL, l); layer >= 0; --layer) {
            auto neighborsWithDist = searchLayer(item.values, entryPoints, efConstruction_, layer);
            
            vector<int> neighbors;
            for (int i = (int)neighborsWithDist.size() - 1; i >= 0; --i) {
                neighbors.push_back(neighborsWithDist[i].second);
            }

            if ((int)neighbors.size() > M_) {
                neighbors.resize(M_);
            }

            nodes_[newIdx].neighbors[layer] = neighbors;
            for (int neighborIdx : neighbors) {
                nodes_[neighborIdx].neighbors[layer].push_back(newIdx);

                if ((int)nodes_[neighborIdx].neighbors[layer].size() > M_) {
                    auto& nList = nodes_[neighborIdx].neighbors[layer];
                    sort(nList.begin(), nList.end(), [&](int a, int b) {
                        return dist(nodes_[neighborIdx].item.values, nodes_[a].item.values) <
                               dist(nodes_[neighborIdx].item.values, nodes_[b].item.values);
                    });
                    nList.resize(M_);
                }
            }

            entryPoints.clear();
            for (int n : neighbors) {
                entryPoints.push_back(n);
            }
        }

        if (l > maxLayer_) {
            maxLayer_ = l;
            entryPoint_ = newIdx;
        }
    }

    vector<pair<VectorItem, float>> search(const vector<float>& query, int k, int efSearch = 50) {
        if (entryPoint_ == -1) return {};

        int currObj = entryPoint_;
        int topL = maxLayer_;

        for (int layer = topL; layer > 0; --layer) {
            currObj = greedyClosest(currObj, query, layer);
        }

        int ef = max(efSearch, k);
        auto candidates = searchLayer(query, {currObj}, ef, 0);

        vector<pair<VectorItem, float>> results;
        for (int i = (int)candidates.size() - 1; i >= 0 && (int)results.size() < k; --i) {
            results.push_back({nodes_[candidates[i].second].item, candidates[i].first});
        }

        return results;
    }

private:
    struct Node {
        VectorItem item;
        int topLayer;
        vector<vector<int>> neighbors;
    };

    vector<Node> nodes_;
    int entryPoint_ = -1;
    int maxLayer_ = -1;
    int M_;
    int efConstruction_;

    int randomLayer() {
        static mt19937 rng(random_device{}());
        static uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double levelMult = 1.0 / log((double)M_);
        return (int)(-log(r) * levelMult);
    }

    float dist(const vector<float>& a, const vector<float>& b) {
        return euclideanDistance(a, b);
    }

    int greedyClosest(int entry, const vector<float>& query, int layer) {
        int current = entry;
        float currentDist = dist(query, nodes_[current].item.values);
        bool improved = true;
        while (improved) {
            improved = false;
            if (layer < (int)nodes_[current].neighbors.size()) {
                for (int neighborIdx : nodes_[current].neighbors[layer]) {
                    float d = dist(query, nodes_[neighborIdx].item.values);
                    if (d < currentDist) {
                        currentDist = d;
                        current = neighborIdx;
                        improved = true;
                    }
                }
            }
        }
        return current;
    }

    vector<pair<float, int>> searchLayer(
        const vector<float>& query,
        const vector<int>& entryPoints,
        int ef,
        int layer
    ) {
        unordered_set<int> visited;
        priority_queue<pair<float, int>, 
                       vector<pair<float, int>>, 
                       greater<pair<float, int>>> candidates;
        
        priority_queue<pair<float, int>> W;

        for (int ep : entryPoints) {
            float d = dist(query, nodes_[ep].item.values);
            visited.insert(ep);
            candidates.push({d, ep});
            W.push({d, ep});
        }

        while (!candidates.empty()) {
            // FIXED: Replaced C++17 structured bindings with direct pair access
            pair<float, int> currentCandidate = candidates.top();
            float cDist = currentCandidate.first;
            int cNode = currentCandidate.second;
            candidates.pop();

            if (W.empty()) break; // Safety check

            pair<float, int> furthestInW = W.top();
            float fDist = furthestInW.first;

            if (cDist > fDist) break;

            if (layer < (int)nodes_[cNode].neighbors.size()) {
                for (int neighbor : nodes_[cNode].neighbors[layer]) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        float d = dist(query, nodes_[neighbor].item.values);

                        if (d < W.top().first || (int)W.size() < ef) {
                            candidates.push({d, neighbor});
                            W.push({d, neighbor});
                            if ((int)W.size() > ef) {
                                W.pop();
                            }
                        }
                    }
                }
            }
        }

        vector<pair<float, int>> result;
        while (!W.empty()) {
            result.push_back(W.top());
            W.pop();
        }
        return result;
    }
};

void printTree(KDNode* node, int depth = 0) {
    if (node == nullptr) return;
    printTree(node->left, depth + 1);
    cout << string(depth * 2, ' ') << node->item.label << "\n";
    printTree(node->right, depth + 1);
}

class VectorDB {
public:
    enum class IndexType { BruteForceIdx, KDTreeIdx, HNSWIdx };

    void insert(const VectorItem& item) {
        allItems_.push_back(item);
        brute_.insert(item);
        hnsw_.insert(item);
        kdDirty_ = true;   // rebuild KDTree lazily on next search (it doesn't support incremental insert)
    }

    bool remove(int id) {
        bool removed = brute_.remove(id);
        allItems_.erase(remove_if(allItems_.begin(), allItems_.end(),
            [id](const VectorItem& it) { return it.id == id; }), allItems_.end());
        kdDirty_ = true;
        // Note: HNSW doesn't support true deletion in this simple version — known limitation, flag it in your README.
        return removed;
    }

    vector<pair<VectorItem, float>> search(const vector<float>& query, int k, IndexType type) {
        switch (type) {
            case IndexType::BruteForceIdx:
                return brute_.search(query, k, euclideanDistance);
            case IndexType::KDTreeIdx:
                rebuildKDTreeIfNeeded();
                return kdtree_.search(query, k);
            case IndexType::HNSWIdx:
                return hnsw_.search(query, k);
        }
        return {};
    }

    size_t size() const { return allItems_.size(); }

private:
    vector<VectorItem> allItems_;
    BruteForce brute_;
    KDTree kdtree_;
    HNSW hnsw_;
    bool kdDirty_ = true;

    void rebuildKDTreeIfNeeded() {
        if (kdDirty_) {
            kdtree_.build(allItems_);
            kdDirty_ = false;
        }
    }
};
int main() {
    
    VectorDB db;
    db.insert({1, "point A", {1.0f, 0.0f, 0.0f}});
    db.insert({2, "point B", {0.0f, 1.0f, 0.0f}});
    db.insert({3, "point C", {0.9f, 0.1f, 0.0f}});
    db.insert({4, "point D", {0.0f, 0.0f, 1.0f}});
    db.insert({5, "point E", {0.8f, 0.2f, 0.0f}});

    vector<float> query = {1.0f, 0.0f, 0.0f};

    auto bf = db.search(query, 3, VectorDB::IndexType::BruteForceIdx);
    auto kd = db.search(query, 3, VectorDB::IndexType::KDTreeIdx);
    auto hn = db.search(query, 3, VectorDB::IndexType::HNSWIdx);

    cout << "BruteForce:\n";
    for (auto& p : bf) cout << "  " << p.first.label << " dist=" << p.second << "\n";
    cout << "KDTree:\n";
    for (auto& p : kd) cout << "  " << p.first.label << " dist=" << p.second << "\n";
    cout << "HNSW:\n";
    for (auto& p : hn) cout << "  " << p.first.label << " dist=" << p.second << "\n";

    return 0;
}