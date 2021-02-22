#pragma once

#include <memory>
#include <vector>

namespace drforest {

    class RandomForest {
    public:
        RandomForest(const std::vector<std::shared_ptr<Tree>> &trees,
                     const arma::mat &X, const arma::vec &y);

        size_t get_num_trees() const {
            return trees_.size();
        }

        const std::vector<std::shared_ptr<Tree>>& get_trees() const {
            return trees_;
        }

        const std::shared_ptr<Tree> get_tree(size_t i) const {
            return trees_.at(i);
        }

        void set_oob_predictions(arma::vec oob_predictions) {
            oob_predictions_ = oob_predictions;
            calculated_oob_ = true;
        }

        const arma::vec& get_oob_predictions() const {
            return oob_predictions_;
        }

        void set_oob_error(double oob_error) {
            oob_error_ = oob_error;
        }

        const double get_oob_error() const {
            return oob_error_;
        }

        arma::vec predict(const arma::mat &X, int num_threads=1) const;

        arma::umat apply(const arma::mat &X, int num_threads=1) const;

        arma::mat estimate_sufficient_dimensions(const arma::mat &X,
                                                 int num_threads=1) const;

    private:
        std::vector<std::shared_ptr<Tree>> trees_;
        const arma::mat& X_;
        const arma::vec& y_;

        bool calculated_oob_;
        double oob_error_;
        arma::vec oob_predictions_;
    };

}  // namespace drforest
