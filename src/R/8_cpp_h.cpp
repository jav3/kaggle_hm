#define ARMA_64BIT_WORD 1

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]] 
// [[Rcpp::depends(RcppProgress)]]

#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
using namespace Rcpp;

#include <Rcpp.h>

//[[Rcpp::export]]
NumericMatrix Rcpp_crossprod(arma::mat& mat, arma::mat& dfm, int topx = 100){
  
  int n_rows = mat.n_rows;
  
  // First topx vals in row will be the topx articles; next topx vals will be the corresponding scores
  NumericMatrix ans(n_rows, topx*2);
  arma::rowvec score;
  Progress p(n_rows, true);
  std::vector<size_t> indices(dfm.n_cols);
  
  for (int i = 0; i < n_rows; i++){
    score = mat.row(i) * dfm;
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + topx, indices.end(),
                      [&](size_t A, size_t B) {
                        return score[A] > score[B];
                      });
    
    for (int j = 0; j < topx; j++){
      ans(i, j) = indices.at(j) + 1;
      ans(i, topx + j) = score.at(indices.at(j));
    }
    p.increment(); 
  }
  
  return ans;
}