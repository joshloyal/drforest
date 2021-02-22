#include "drforest.h"

namespace drforest {

    RandomForest::RandomForest(const std::vector<std::shared_ptr<Tree>> &trees,
                               const arma::mat &X, const arma::vec &y) :
            trees_(trees), X_(X), y_(y), calculated_oob_(false) {}

    arma::vec RandomForest::predict(const arma::mat &X, int num_threads) const {
        // determine number of trees to predict in parallel
        num_threads = get_num_threads(num_threads);

        int t;
        arma::vec preds(X.n_rows, arma::fill::zeros);
        #pragma omp parallel num_threads(num_threads) private(t)
        {
            arma::vec preds_local(X.n_rows, arma::fill::zeros);

            #pragma omp for schedule(static)
            for(t = 0; t < get_num_trees(); t++) {
                preds_local += trees_.at(t)->predict(X);
            }

            #pragma omp critical
            preds += preds_local;
        }

        preds /= (double) get_num_trees();

        return preds;
    }

    arma::umat RandomForest::apply(const arma::mat &X, int num_threads) const {
        // determine number of trees to apply in parallel
        num_threads = get_num_threads(num_threads);

        int t;
        size_t num_trees = get_num_trees();
        arma::umat leafs(X.n_rows, num_trees, arma::fill::zeros);
        #pragma omp parallel for schedule(static) num_threads(num_threads) private(t)
        for(t = 0; t < num_trees; t++) {
            leafs.col(t) = trees_.at(t)->apply(X);
        }

        return leafs;
    }

    arma::mat RandomForest::estimate_sufficient_dimensions(
            const arma::mat &X, int num_threads) const {
        // determine number of trees to apply in parallel
        num_threads = get_num_threads(num_threads);

        std::pair<arma::mat, arma::mat> whiten_results =
            drforest::math::whiten(X);
        arma::mat Z = whiten_results.first;
        arma::mat R = whiten_results.second;

        int t;
        arma::mat M(X.n_cols, X.n_cols, arma::fill::zeros);
        #pragma omp parallel num_threads(num_threads) private(t)
        {
            arma::mat M_local(X.n_cols, X.n_cols, arma::fill::zeros);

            #pragma omp for schedule(static)
            for(t = 0; t < get_num_trees(); t++) {
                M_local += trees_.at(t)->estimate_M(X, Z);
            }

            #pragma omp critical
            M += M_local;
        }

        M /= (double) get_num_trees();

        // eigen decomposition
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, M);

        // transform eigenvalues back to feature space and
        // normalise to unit norm
        arma::mat directions;
        if (X.n_rows >= X.n_cols) {
            directions = arma::solve(
                arma::trimatu(std::sqrt(X.n_rows) * R), arma::fliplr(eigvec));
        } else {
            directions = arma::solve(
                std::sqrt(X.n_rows) * R, arma::fliplr(eigvec));
        }
        directions = arma::normalise(directions).t();

        return directions;
    }

}  // namespace drforest
