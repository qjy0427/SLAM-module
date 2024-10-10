#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include "bow_index.hpp"
#include "keyframe.hpp"
#include "map_point.hpp"

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
  archive(mapKf.mapId, mapKf.kfId);
}

template<class Archive>
void serialize(Archive &archive, slam::MapId &mapId) {
  archive(mapId.v);
}


template<class Archive>
void serialize(Archive &archive, slam::KfId &kfId) {
  archive(kfId.v);
}

template<class Archive>
void serialize(Archive &archive, slam::MapPoint &mapPoint) {
    archive(mapPoint.id, mapPoint.trackId, mapPoint.status, mapPoint.position,
        mapPoint.norm, mapPoint.minViewingDistance, mapPoint.maxViewingDistance,
        mapPoint.descriptor, mapPoint.observations, mapPoint.referenceKeyframe,
        mapPoint.color);
}

template<class Archive>
void serialize(Archive &archive, slam::MpId &mpId) {
    archive(mpId.v);
}

template<class Archive>
void serialize(Archive &archive, slam::KpId &kpId) {
    archive(kpId.v);
}

template<class Archive>
void serialize(Archive &archive, slam::TrackId &trackId) {
    archive(trackId.v);
}

template<class Archive>
void serialize(Archive &archive, slam::Keyframe &keyframe) {
    archive(keyframe.shared, keyframe.id, keyframe.previousKfId,
        keyframe.nextKfId, keyframe.keyPointToTrackId, keyframe.mapPoints,
        keyframe.keyPointDepth, keyframe.poseCW, keyframe.smoothPoseCW,
        keyframe.uncertainty, keyframe.t, keyframe.hasFullFeatures);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options,
int _MaxRows, int _MaxCols>
void serialize(Archive& archive,
          const Eigen::Matrix<_Scalar, _Rows, _Cols,
            _Options, _MaxRows, _MaxCols>& matrix) {
    // Save shape
    const std::int32_t rows = matrix.rows();
    const std::int32_t cols = matrix.cols();
    archive(rows, cols);

    // Save data
    std::vector<_Scalar> data(matrix.data(), matrix.data() + matrix.size());
    archive(data);
}

template<class Archive, typename T>
void serialize(Archive &archive, cv::Vec<T, 3> &vec) {
    archive(vec[0], vec[1], vec[2]);
}

template<class Archive>
void serialize(Archive &archive, slam::KeyframeShared &keyframeShared) {
    archive(keyframeShared.camera, keyframeShared.keyPoints,
        keyframeShared.featureSearch, keyframeShared.imgDbg,
        keyframeShared.colors, keyframeShared.stereoPointCloud,
        keyframeShared.stereoPointCloudColor, keyframeShared.bowVec,
        keyframeShared.bowFeatureVec);
}

template<class Archive>
void serialize(Archive &archive, slam::KeyPoint &keyPoint) {
    archive(keyPoint.pt, keyPoint.angle, keyPoint.octave,
        keyPoint.bearing, keyPoint.descriptor);
}

template<class Archive>
void serialize(Archive &archive, tracker::Feature::Point &point) {
    archive(point.x, point.y);
}

}   // namespace cereal
