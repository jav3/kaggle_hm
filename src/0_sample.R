library(data.table)
library(dplyr)

setwd("~/Downloads/Kaggle/hm")

articles = fread("data/articles.csv")
customers = fread("data/customers.csv")
trans = fread("data/transactions_train.csv")

# Map customer_id within trans
cust_indices = trans[, list(customer_id = unique(customer_id))]
cust_indices[, cust_idx := 1:.N]
trans_2 = merge(trans, cust_indices, by = "customer_id", all = T)
trans_2[, customer_id := NULL]
fwrite(trans_2, "data/trans.csv" )
fwrite(cust_indices, "data/customer_indices.csv")

# Map customer_id and postal_code within customers
pc_indices = customers[, list(postal_code = unique(postal_code))]
pc_indices[, pc_idx := 1:.N]
customers_2 = merge(customers, cust_indices, by = "customer_id", all = T)
customers_2 = merge(customers_2, pc_indices, by = "postal_code", all = T)
customers_2[, customer_id := NULL]
customers_2[, postal_code := NULL]
fwrite(customers_2, "data/cust.csv" )
fwrite(pc_indices, "data/pc_indices.csv" )

# Sample
set.seed(100)

pct = 0.025
cust_idx_samp = sample(cust_indices$cust_idx, floor(pct*nrow(cust_indices)), replace = F)

cust_indices_samp = cust_indices[cust_idx %in% cust_idx_samp]
trans_samp = trans_2[cust_idx %in% cust_idx_samp]
customers_samp = customers_2[cust_idx %in% cust_idx_samp]
pc_indices_samp = pc_indices[pc_idx %in% customers_samp$pc_idx]

fwrite(cust_indices_samp, "data/sample/customer_indices.csv")
fwrite(trans_samp, "data/sample/trans_samp.csv")
fwrite(customers_samp, "data/sample/customers_samp.csv")
fwrite(pc_indices_samp, "data/sample/pc_indices_samp.csv")

