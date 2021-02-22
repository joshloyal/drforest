#pragma once

namespace drforest {

    class Categorical {
        public:
            Categorical(uint seed=42);

            // set weights of the categorical
            uint draw(arma::vec &weights);
        private:
            std::mt19937_64 random_state_;
    };
}
