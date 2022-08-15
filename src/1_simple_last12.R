library(data.table)
library(dplyr)

# Read data ---------------------------------------------------------------

setwd("~/Downloads/Kaggle/hm")

articles = fread("data/articles.csv")
customers = fread("data/customers.csv")
trans = fread("data/transactions_train.csv")

# Latest date
max(trans$t_dat) # 2020-09-22

# Number of customers
length(unique(customers$customer_id)) # 1371980


# Build simple submission -------------------------------------------------
# Use last 12 purchases
# Do not fill if not enough purchases

setkey(trans, customer_id)
trans = trans[order(customer_id, -t_dat)]
trans[, article_id := paste0("0", article_id)]

last12 = trans[, .(customer_id, article_id)][, .(prediction = paste0(head(article_id, 12), sep = " ", collapse = " ")), 
                                                 by = customer_id]
submission = merge(customers[, .(customer_id)], last12, by = "customer_id", all.x = T)
data.table::fwrite(submission, file = "submission_last12.csv.gz")
# Scores 0.0188 on the leaderboard
