#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/filesystem.hpp>

#include "drforest.h"

namespace drforest {
namespace testing {

arma::vec read_vector_from_file(std::string filename) {
    std::vector<double> result;

    // Open input file
    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.good()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream line_stream(line);
        double element;
        if (!(line_stream >> element)) {
            break;  // Error reading line.
        }
        result.push_back(element);
    }
    input_file.close();

    return arma::conv_to<arma::vec>::from(result);
}


std::string find_local_path(const std::string &file_or_dir_name) {
    std::vector<boost::filesystem::path> candidates;
    candidates.push_back(boost::filesystem::path(file_or_dir_name));
    candidates.push_back(boost::filesystem::path("./data/" + file_or_dir_name));
    candidates.push_back(boost::filesystem::path("../tests/data/" + file_or_dir_name));

    for (const auto& relative_path : candidates) {
        auto absolute_path = boost::filesystem::absolute(relative_path);
        if(boost::filesystem::exists(absolute_path)) {
            std::string path = boost::filesystem::canonical(absolute_path).string();
            return path;
        }
    }

    return "";
}

std::pair<arma::mat, arma::vec> load_athletes() {
    std::string data_name{"athletes_X.txt"};
    auto found_file = drforest::testing::find_local_path(data_name);
    arma::mat X;
    X.load(found_file, arma::csv_ascii);

    data_name = "athletes_y.txt";
    found_file = drforest::testing::find_local_path(data_name);
    arma::vec y = drforest::testing::read_vector_from_file(found_file);

    return {X, y};
}


arma::mat load_athletes_directions() {
    std::string data_name{"athletes_directions.txt"};
    auto found_file = drforest::testing::find_local_path(data_name);
    arma::mat directions;
    directions.load(found_file, arma::csv_ascii);

    return directions.t();
}


std::pair<arma::mat, arma::vec> load_quadratic() {
    std::string data_name{"quadratic_X.txt"};
    auto found_file = drforest::testing::find_local_path(data_name);
    arma::mat X;
    X.load(found_file, arma::csv_ascii);

    data_name = "quadratic_y.txt";
    found_file = drforest::testing::find_local_path(data_name);
    arma::vec y = drforest::testing::read_vector_from_file(found_file);

    return {X, y};
}


arma::mat load_quadratic_directions() {
    std::string data_name{"quadratic_directions.txt"};
    auto found_file = drforest::testing::find_local_path(data_name);
    arma::mat directions;
    directions.load(found_file, arma::csv_ascii);

    return directions.t();
}

std::pair<arma::mat, arma::vec> make_simple_split() {
    arma::arma_rng::set_seed(123);

    arma::mat X(500, 10, arma::fill::randn);
    arma::vec beta{1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    arma::vec Z = X * beta;

    arma::vec y(500);
    double cut = Z(200);

    arma::uvec threshold = arma::find(Z > cut);
    y.elem(threshold) = 10
        + 2 * arma::vec(threshold.n_rows, arma::fill::randn);

    threshold = arma::find(Z <= cut);
    y.elem(threshold) = -10 +
        2 * arma::vec(threshold.n_rows, arma::fill::randn);

    return {X, y};
}

std::pair<arma::mat, arma::vec> make_checkerboard() {
    arma::arma_rng::set_seed(123);

    arma::mat X(500, 10, arma::fill::randn);
    arma::vec beta1{1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    arma::vec Z1 = X * beta1;

    arma::vec beta2{0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
    arma::vec Z2 = X * beta2;

    arma::vec y(500);
    double cut1 = Z1(200);
    double cut2 = Z2(30);

    arma::uvec threshold = arma::find(Z1 > cut1 && Z2 > cut2);
    y.elem(threshold) = 10
        + 2 * arma::vec(threshold.n_rows, arma::fill::randn);

    threshold = arma::find(Z1 > cut1 && Z2 <= cut2);
    y.elem(threshold) = 20
        + 2 * arma::vec(threshold.n_rows, arma::fill::randn);

   threshold = arma::find(Z1 <= cut1 && Z2 <= cut2);
    y.elem(threshold) = -10
        + 2 * arma::vec(threshold.n_rows, arma::fill::randn);

    threshold = arma::find(Z1 <= cut1 && Z2 > cut2);
    y.elem(threshold) = -20
        + 2 * arma::vec(threshold.n_rows, arma::fill::randn);

    return {X, y};
}

}  // namespace testing
}  // namespace drforest
