#ifndef SLAM_BOW_INDEX_HPP
#define SLAM_BOW_INDEX_HPP

#include <list>
#include <memory>
#include <vector>

#include <cereal/types/list.hpp>
#include <cereal/types/vector.hpp>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

#include "id.hpp"
#include "key_point.hpp"
#include "../odometry/parameters.hpp"

namespace slam {

class Keyframe;
class MapDB;

using BowVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
using Atlas = std::vector<MapDB>;

struct MapKf {
    MapId mapId;
    KfId kfId;
};

bool operator == (const MapKf &lhs, const MapKf &rhs);
bool operator < (const MapKf &lhs, const MapKf &rhs);

struct BowSimilar {
    MapKf mapKf;
    float score;
};

class BowIndex {
public:
    BowIndex(const odometry::ParametersSlam &parameter);

    void add(const Keyframe &keyframe, MapId mapId);

    void remove(MapKf mapKf);

    void transform(
        const KeyPointVector &keypoints,
        DBoW2::BowVector &bowVector,
        DBoW2::FeatureVector &bowFeatureVector
    );

    // Get all keyframes similar to a query keyframe.
    std::vector<BowSimilar> getBowSimilar(const MapDB &mapDB, const Atlas &atlas, const Keyframe &kf);

    template <class Archive>
    void serialize(Archive& ar) {
        ar(index, tmp);
    }

    BowIndex& operator=(const BowIndex& other) {
        if (this == &other) {
            return *this;
        }
        index = other.index;
        tmp = other.tmp;
        return *this;
    }

private:
    const odometry::ParametersSlam &parameters;

    // Called inverse index in the DBoW paper.
    std::vector<std::list<MapKf>> index;
    struct Workspace {
        std::vector<cv::Mat> descVector, cvMatStore;

        template<class Archive>
        void serialize(Archive & archive) {
            archive(descVector, cvMatStore);
        }
    } tmp;

    BowVocabulary bowVocabulary;
};

}  // namespace slam

namespace cereal {

template<class Archive>
void serialize(Archive &archive, cv::Mat &mat) {
    int rows, cols, type;
    bool continuous;

    // 序列化 Mat 的元数据
    if (Archive::is_saving::value) {
        rows = mat.rows;
        cols = mat.cols;
        type = mat.type();
        continuous = mat.isContinuous();
    }

    archive(rows, cols, type, continuous);

    if (Archive::is_loading::value) {
        mat.create(rows, cols, type);
    }

    // 序列化 Mat 的数据
    if (continuous) {
        const size_t data_size = rows * cols * mat.elemSize();
        archive(binary_data(mat.ptr(), data_size));
    } else {
        const size_t row_size = cols * mat.elemSize();
        for (int i = 0; i < rows; i++) {
            archive(binary_data(mat.ptr(i), row_size));
        }
    }
}

template<class Archive>
void serialize(Archive &archive, slam::MapKf &mapKf) {
    archive(mapKf.mapId.v, mapKf.kfId.v);
}

}   // namespace cereal

#endif  // SLAM_BOW_INDEX_HPP
