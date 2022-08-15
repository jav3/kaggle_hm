library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
data.table::setDTthreads(4)
library(cmfrec)
library(Matrix)
library(MatrixExtra)
library(recometrics)

setwd("~/Downloads/Kaggle/hm")
source("./src/8_arrow_h.R")
Rcpp::sourceCpp("./src/8_cpp_h.cpp")

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = articles %>% collect %>% as.data.table
customers_data = customers %>% collect %>% as.data.table

num_articles = articles %>% pull(aid) %>% n_distinct
num_customers = customers %>% pull(cid) %>% n_distinct
aid_list = articles %>% pull(aid)
cid_list = customers %>% pull(cid)

# Dynamic attr
articles_dy = readRDS("data/t8/articles_dy.rds")
setDT(articles_dy)
customers_dy = readRDS("data/t8/customers_dy.rds")
setDT(customers_dy)

articles_dfm = readRDS("data/t8/articles_dfm.rds")
articles_lsa = readRDS("data/t8/articles_lsa100.rds")

curr_yr = 2020
wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(37, 38, 39) # Test on these weeks - the last one is for Kaggle

# Results
res = vector("list", length(wks_test))

# Construct matrix --------------------------------------------------------


for (i in seq_len(length(wks_test))){
  
  
  week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
  wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
  curr_wk = wks_test[i]
  
  prior_purchases = transactions %>% 
    filter(yr == curr_yr, wk %in% wks) %>%
    select(cid, aid) %>%
    collect() %>%
    unique
  setDT(prior_purchases)
  
  ui_prior = sparseMatrix(i = prior_purchases$cid, j = prior_purchases$aid, x = 1, 
                          dims = c(num_customers, num_articles))
  dimnames(ui_prior) = list(cid_list, aid_list)
  
  # Keep only observed cid and aid
  obs_cid = which(rowSums(ui_prior) > 0)
  obs_aid = which(colSums(ui_prior) > 0)
  
  ui_prior_f = ui_prior[obs_cid, obs_aid]
  ui_prior_f = as.coo.matrix(ui_prior_f)
  
  # User information
  U = customers_dy[wk_predict == curr_wk]
  U = merge(data.table(cid = cid_list), U, by = "cid", all.x = T)
  U = merge(U, customers_data[, .(cid, age, FN, Active)], by = "cid", all.x = T)
  U[, unseen_c := is.na(days_since_last_seen_c)*1]
  set(U, which(is.na(U$days_since_last_seen_c)), "days_since_last_seen_c", -1)
  set(U, which(is.na(U$average_channel)), "average_channel", median(U$average_channel, na.rm = T))
  set(U, which(is.na(U$age)), "age", median(U$age, na.rm = T))
  U[, FN := as.numeric(as.character(FN))]
  U[, Active := as.numeric(as.character(Active))]
  set(U, which(is.na(U$FN)), "FN", 0)
  set(U, which(is.na(U$Active)), "Active", 0)
  setorder(U, cid)
  U_all = U[, grp := .GRP, by = .(average_channel, days_since_last_seen_c, age, FN, Active)][order(cid)]
  U = U_all[, .(average_channel, days_since_last_seen_c, age, FN, Active)] %>% as.matrix() %>% as("dgTMatrix")
  #U = apply(U, 2, function(x) (x - mean(x))/sd(x))
  U_f = U[obs_cid,]
  U_distinct = U_all[, head(.SD, 1), by = .(grp)][, .(average_channel, days_since_last_seen_c, age, FN, Active)] %>% 
    as.matrix() %>%
    as("dgTMatrix")
  #U_distinct = apply(U_distinct, 2, function(x) (x - mean(x))/sd(x))
  
  
  
  # Article information
  a_data = model.matrix(~ . - 1, data = articles_data[, .(perceived_colour_master_name, index_name, index_group_name, section_name, garment_group_name)])
  a_data = Matrix(a_data, sparse = T)
  a_dy = articles_dy[order(aid)][wk_predict == curr_wk, .(days_since_last_seen, days_on_market, price, stock_count)]
  a_dy[, unseen := is.na(days_on_market)*1]
  set(a_dy, which(is.na(a_dy$days_since_last_seen)), "days_since_last_seen", -1)
  set(a_dy, which(is.na(a_dy$days_on_market)), "days_on_market", -1)
  set(a_dy, which(is.na(a_dy$price)), "price", median(a_dy$price, na.rm = T))
  set(a_dy, which(is.na(a_dy$stock_count)), "stock_count", 0)
  a_dy = as.matrix(a_dy)
  # a_dy = apply(a_dy, 2, function(x) (x - mean(x))/sd(x))
  a_dy = Matrix(a_dy, sparse = T)
  A = cbind(as(articles_dfm, "Matrix"), a_data, a_dy)
  A_f = A[obs_aid,] %>% as("dgTMatrix")
  
  # Train -------------------------------------------------------------------
  
  if (F){
    
    reco_split <- create.reco.train.test(
      ui_prior_f,
      users_test_fraction = NULL,
      max_test_users = round(0.05*nrow(ui_prior_f)),
      items_test_fraction = 0.3,
      seed = 123
    )
    
    grid_search = expand.grid(k = c(50, 100, 200), alpha = c(40), lambda = c(1))
    
    grid_search_res = lapply(1:nrow(grid_search), function(x){
      mod1 = CMF_implicit(as.coo.matrix(reco_split$X_rem), k = grid_search[["k"]][x],
                          center_U = TRUE, center_I = TRUE, alpha = grid_search[["alpha"]][x],
                          lambda = grid_search[["lambda"]][x], U = U_f[-reco_split$users_test,], I = A_f)
      res = calc.reco.metrics(reco_split$X_train, reco_split$X_test, k = 12, 
                              A = t(cmfrec::factors(mod1, reco_split$X_train, U = U_f[reco_split$users_test,])),
                              item_biases = mod1$matrices$item_bias,
                              B = mod1$matrices$B, all_metrics = F, average_precision = T) %>%
        sapply(mean, na.rm = T)
      return (res)
    })
    
    grid_search_res_2 = cbind(grid_search, rbindlist(lapply(grid_search_res, function(x) as.data.frame(as.list(x)))))
    setDT(grid_search_res_2)
    grid_search_res_2[order(-ap_at_12)]
    
    mod2 = MostPopular(X = as.coo.matrix(reco_split$X_rem), implicit = T) 
    calc.reco.metrics(reco_split$X_train, reco_split$X_test,
                      A = NULL, B = NULL,
                      item_biases = mod2$matrices$item_bias,
                      k = 12, all_metrics = F, average_precision = T) %>%
      sapply(mean)
    
  }
  
  
  # Re-train on all
  mod1 = CMF_implicit(as.coo.matrix(ui_prior_f), k = 100,
                      center_U = TRUE, center_I = TRUE, alpha = 40,
                      lambda = 1, U = U_f, I = A_f, seed = 123)
  
  # Predict -----------------------------------------------------------------
  
  num_recs = 12
  
  # Existing users
  ui_prior_f_preds = Rcpp_crossprod(t(mod1$matrices$A), mod1$matrices$B, topx = num_recs)
  cbf = data.table(cid = cid_list[obs_cid], 
                   preds = asplit(ui_prior_f_preds[,1:num_recs], 1),
                   scores = asplit(ui_prior_f_preds[,(num_recs + 1):(2*num_recs)], 1)
  )
  cbf = cbf[, .(item = unlist(preds), score = unlist(scores)), by = .(cid)]
  
  # Unseen users - get distinct first
  preds = pblapply(seq_len(nrow(U_distinct)), function(x){
    res = topN_new(mod1,
                   U = U_distinct[x, , drop = T], n = num_recs, output_score = T) %>%
      as.data.table
    merge(U_all[grp == x, .(k = 1, cid)],
          res[, .(k = 1, item, score)],
          all = T, by = "k", allow.cartesian = T)[,k := NULL]
  })
  preds = rbindlist(preds)
  preds = preds[!(cid %in% cid_list[obs_cid])]
  
  all = rbindlist(list(cbf %>% mutate(observed = 1), preds %>% mutate(observed = 0)))
  all[, aid := aid_list[obs_aid][item]]
  
  all[, item := NULL]
  all[, wk_predict := wks_test[i]]
  res[[i]] = all
  
  if (0){
    checkPR(all[order(cid, -score)][, head(.SD, 3), by = .(cid)][, .(cid, aid)], curr_yr = 2020, curr_wk = curr_wk)
    checkPR(customers_data[, .(aid = unlist(topN_pop)), by = .(cid)], curr_yr = 2020, curr_wk = curr_wk)
  }
}


mf = rbindlist(res)

saveRDS(mf, "data/t8/mf.rds")
