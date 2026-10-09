#include "mnist.h"

#include <fstream>
#include <stdexcept>

static std::uint32_t read_u32_be(std::ifstream& file) {
    unsigned char bytes[4];

    if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
        throw std::runtime_error("Could not read IDX header");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

MnistData load_mnist(const std::string& image_path,
                     const std::string& label_path) {
    std::ifstream images_file(image_path, std::ios::binary);
    std::ifstream labels_file(label_path, std::ios::binary);

    if (!images_file) {
        throw std::runtime_error("Could not open image file: " + image_path);
    }
    if (!labels_file) {
        throw std::runtime_error("Could not open label file: " + label_path);
    }

    const std::uint32_t image_magic = read_u32_be(images_file);
    const std::uint32_t image_count = read_u32_be(images_file);
    const std::uint32_t rows = read_u32_be(images_file);
    const std::uint32_t cols = read_u32_be(images_file);

    const std::uint32_t label_magic = read_u32_be(labels_file);
    const std::uint32_t label_count = read_u32_be(labels_file);

    if (image_magic != 2051) {
        throw std::runtime_error("Unexpected MNIST image-file magic number");
    }
    if (label_magic != 2049) {
        throw std::runtime_error("Unexpected MNIST label-file magic number");
    }
    if (image_count != label_count) {
        throw std::runtime_error("Image count and label count do not match");
    }

    MnistData data;
    data.rows = rows;
    data.cols = cols;
    data.images.resize(static_cast<std::size_t>(image_count) * rows * cols);
    data.labels.resize(label_count);

    if (!images_file.read(reinterpret_cast<char*>(data.images.data()),
                          static_cast<std::streamsize>(data.images.size()))) {
        throw std::runtime_error("Could not read all image pixels");
    }
    if (!labels_file.read(reinterpret_cast<char*>(data.labels.data()),
                          static_cast<std::streamsize>(data.labels.size()))) {
        throw std::runtime_error("Could not read all labels");
    }

    return data;
}