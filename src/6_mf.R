library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Matrix)

setwd("~/Downloads/Kaggle/hm")

# Read data ---------------------------------------------------------------
articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

curr_yr = 2020
wk_start = 34
wk_end = 36:38 # We will generate from wk_start to wk_end (for each case), and save the data

# Results
res = vector(mode = "list", length = length(wk_end))
names(res) = wk_end

# Loop through cases now
for (i in wk_end){
  
  
  # Construct matrix --------------------------------------------------------
  num_articles = articles %>% pull(aid) %>% n_distinct
  num_customers = customers %>% pull(cid) %>% n_distinct
  
  idx = transactions %>% 
    filter(yr == curr_yr, wk %in% wk_start:i) %>%
    select(cid, aid) %>%
    collect() %>%
    unique
  
  ui = sparseMatrix(idx$cid, idx$aid, dims = c(num_customers, num_articles))
  
  ui_norm = Matrix::Diagonal(x = 1 / sqrt(Matrix::rowSums(ui^2)))  %*% ui
  
  um = tcrossprod(ui_norm)
  
  
  # Number of linked "potentially similar" customers by customer
  n_relationships_by_cust = diff(um@p)
  relations_expanded = data.table("cid" = rep(seq_along(n_relationships_by_cust), n_relationships_by_cust),
                                  "linked_cid" = um@i + 1,
                                  "similarity" = um@x)
  
  # Cleanup
  rm(list = c("ui", "ui_norm", "um", "idx", "n_relationships_by_cust"))
  gc()
  
  # Keep only top 13 (1 will be the customer linked to themself)
  
  # Order in descending order of similarity by cid
  setorder(relations_expanded, cid, -similarity)
  
  grpn = uniqueN(relations_expanded$cid)
  pb <- txtProgressBar(min = 0, max = grpn, style = 3)
  
  relations_expanded = relations_expanded[relations_expanded[, {setTxtProgressBar(pb, .GRP); .I[1:min(.N, 13)];}, by = .(cid)]$V1]
  
  # Remove cases of customer linked to themselves
  relations_expanded = relations_expanded[cid != linked_cid]
  
  # Get corresponding articles
  articles_purchased = transactions %>% 
    filter(yr == curr_yr, wk %in% wk_start:i) %>%
    select(cid, aid) %>%
    collect() %>%
    unique
  relations_expanded = merge(relations_expanded, articles_purchased, by.x = "linked_cid", by.y = "cid", allow.cartesian = T)
  
  # Summarize article similarity across (customer, aid)
  res[[as.character(i)]] = relations_expanded[, .(similarity = sum(similarity, na.rm = T)), by = .(cid, aid)]
  
  gc()
}

saveRDS(res, "data/ucf.rds")
