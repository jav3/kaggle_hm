library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
library(arules)
library(recommenderlab)
library(ggplot2)
data.table::setDTthreads(1)

setwd("~/Downloads/Kaggle/hm")

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = articles %>% collect
setDT(articles_data)

curr_yr = 2020
wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(35, 36, 37, 38, 39) # Test on these weeks - the last one is for Kaggle

i = 1

week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
half_life = 5 # 5 days assumption

# Prior purchases
purchases = transactions %>%
  filter(yr == curr_yr, wk %in% wks) %>% 
  collect
setDT(purchases)

# Curr purchases
curr_purchases = transactions %>% 
  filter(yr == curr_yr, wk == wks_test[i]) %>% 
  select(aid, cid) %>% 
  distinct %>% 
  mutate(curr_purchase = 1) %>% 
  collect() %>% 
  as.data.table


# User-item matrix
# Transactions object
# purchases[, aid := as.character(aid)]
# purchases[, cid := as.character(cid)]
set.seed(100)
trans_list = purchases[, .(list(unique(aid))), by = .(cid)]
trans_list_2 = transactions(trans_list$V1)

imat <- as(trans_list_2, "itemMatrix")

rules <- apriori(
  imat,
  parameter = list(
    supp = 0.00002,
    conf = 0.1,
    target = "rules"
  )
)
rules

recommend <- function(rules, newdata) {
  # Which transactions in newdata match
  # the LHS of which rule?
  lhs_match <- is.superset(
    newdata,
    lhs(rules),
    sparse = TRUE
  )
  
  # For which (item, transaction) pairs is
  # the RHS also matched?
  rec <- tcrossprod(lhs_match, rhs(rules)@data)
  rec_stats = lhs_match %*% data.matrix(rules@quality[, c("support", "confidence")])
  
  # Make sure the row/column names
  # are the same
  rownames(rec) <- rownames(newdata)
  colnames(rec) <- colnames(newdata)
  
  # return an itemMatrix
  return(list(rec = as(t(rec), "itemMatrix"), rec_stats = rec_stats))
}

rec_run <- recommend(rules, imat)
rec = rec_run$rec
rec_stats = rec_run$rec_stats

rec_idx = data.table(which(rec@data, arr.ind = T))
rec_idx[, support := rec_stats[,1][row]]
rec_idx[, confidence := rec_stats[,2][row]]
rec_idx[, aid := as.numeric(rec@itemInfo$labels[row])]
rec_idx[, cid := as.numeric(trans_list$cid)[col]]

rec_idx = merge(rec_idx, purchases[, .(aid, cid, purchased = 1)] %>% unique, 
                by = c("cid", "aid"), all.x = T, all.y = F)
rec_idx = rec_idx[is.na(purchased)]
rec_idx[, pred := 1]

curr_purchases_with_pred = merge(curr_purchases, rec_idx, by = c("cid", "aid"), all.x = T, all.y = T)
curr_purchases_with_pred[pred == 1 & purchased == 1]

intersect(unique(purchases$cid), unique(curr_purchases$cid))[1:10]
cid_c = 632159
purchases[cid == cid_c]
curr_purchases[cid == cid_c]
rec_idx[cid == cid_c]

if (0){
itemInfo(trans_list) = merge(itemInfo(trans_list), 
                             articles_data[, .(aid, index_group_name)],
                             by.x = "labels", by.y = "aid", all.x = T)
trans_list_agg = addAggregate(trans_list, "index_group_name")
trans_list_2 = as(trans_list_agg, "binaryRatingMatrix")

# Eval scheme
scheme <- trans_list_2 %>% 
  evaluationScheme(method = "cross-validation",
                   k      = 3, 
                   train  = 0.5,  
                   given  = -1)

algorithms <- list(
  "association rules" = list(name  = "AR", param = list(supp = 0.00001, conf = 0.0001)),
  "random items"      = list(name  = "RANDOM",  param = NULL),
  "popular items"     = list(name  = "POPULAR", param = NULL)
)

n_vals = c(12, 15, 20)
results <- recommenderlab::evaluate(scheme, 
                                    algorithms, 
                                    type  = "topNList", 
                                    n     = n_vals
)

avg_conf_matr <- function(results) {
  tmp <- results %>%
    getConfusionMatrix()  %>%  
    as.list()
  as.data.frame(Reduce("+",tmp) / length(tmp)) %>% 
    mutate(n = n_vals) %>%
    select('n', 'precision', 'recall', 'TPR', 'FPR') 
}

results_tbl <- lapply(results, avg_conf_matr)
results_tbl = rbindlist(results_tbl, idcol = c("alg" = names(results_tbl)))
setnames(results_tbl, 1, "name")

results_tbl %>%
  ggplot(aes(recall, precision, 
             colour = forcats::fct_reorder2(as.factor(name),  
                                            precision, recall))) +
  geom_line() +
  geom_label(aes(label = n))  +
  labs(title = "Precision-Recall curves", colour = "Model") +
  theme_grey(base_size = 14)

rec = Recommender(trans_list_2, method = "AR", parameter = list(supp = 0.00001, conf = 0.0001))

pred = predict(rec, trans_list_2, n = 12)
}
