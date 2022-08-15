library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
library(quanteda)
library(Matrix)
library(quanteda.textstats)
library(quanteda.textmodels)

setwd("~/Downloads/Kaggle/hm")
Rcpp::sourceCpp("src/8_cpp_h.cpp")


# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = collect(articles)
setDT(articles_data)
articles_data[, desc_idx := .GRP, by = .(detail_desc)]

num_articles = articles %>% pull(aid) %>% n_distinct
num_customers = customers %>% pull(cid) %>% n_distinct
aid_list = articles %>% pull(aid)
cid_list = customers %>% pull(cid)

# Article price -----------------------------------------------------------
# trans = transactions %>% 
#   select(t_dat, aid, cid, price) %>%
#   collect
# setDT(trans)
# setorder(trans, aid, -t_dat)
# articles_price = trans[, .SD[1:min(.N, 10)], by = .(aid)]
# articles_price = articles_price[, .(price = median(price)*590), by = .(aid)]

# Similar articles --------------------------------------------------------

art_corpus = corpus(articles_data[, paste(prod_name, product_type_name, product_group_name, 
                                          graphical_appearance_name, colour_group_name, perceived_colour_value_name, 
                                          perceived_colour_master_name, department_name, index_name, index_group_name, 
                                          section_name, garment_group_name, detail_desc)])
docvars(art_corpus, "aid") = articles_data$aid

tokens = tokens(art_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)

art_dfm = dfm(tokens)
art_dfm = dfm_remove(art_dfm, stopwords("en"))
art_dfm = dfm_wordstem(art_dfm)
art_dfm = dfm_trim(art_dfm, min_docfreq = 1)
art_dfm = dfm_tfidf(art_dfm)
dim(art_dfm)
saveRDS(art_dfm, "data/t8/articles_dfm.rds")

# Article similarity computation
batch_split = (seq(nrow(art_dfm)) %/% 2000) + 1
res = vector(mode = "list", length(unique(batch_split)))
pb <- txtProgressBar(min = 0, max = length(res), style = 3)
for (i in unique(batch_split)){
  setTxtProgressBar(pb, i)
  res[[i]] = textstat_simil(art_dfm, art_dfm[which(batch_split == i),], margin = "documents", method = "cosine", min_simil = 0.5)
}

res_2 = unlist(lapply(res, as.list), recursive = F)
res_2 = Map(function(x, nm) data.table(aid = nm, linked_aid = 
                                         as.numeric(gsub("text", "", names(x))), score = x), 
            res_2, 
            as.numeric(gsub("text", "", names(res_2))))

res_2 = rbindlist(res_2, use.names = T, fill = T)

if (F){
  saveRDS(res_2, "data/t8/articles_similarity.rds")
}

# Dimension reduction
art_lsa = textmodel_lsa(art_dfm, nd = 100, margin = "features")
art_lsa = art_lsa$docs
saveRDS(art_lsa, "data/t8/articles_lsa100.rds")

# User similarity ---------------------------------------------------------
# 
# curr_yr = 2020
# wks_train = 2 # Train for 2 weeks prior to wks_test
# wks_test = c(38, 39) # Test on these weeks - the last one is for Kaggle
# i = 1
# week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
# wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
# half_life = 5 # 5 days assumption
# 
# purchases = transactions %>%
#   filter(yr == curr_yr, wk %in% wks) %>% 
#   collect
# setDT(purchases)
# 
# purchases[, days_to_end_of_train := as.numeric(week_end_date - t_dat)]
# purchases[, popularity := 1/2^(days_to_end_of_train/half_life)]
# purchases = purchases[, .(relevance_ac = popularity), by = .(aid, cid)]
# purchases[, relevance_ac := relevance_ac/sum(relevance_ac), by = .(cid)]
# 
# ui_mat = sparseMatrix(i = purchases$cid, j = purchases$aid, x = purchases$relevance_ac, dims = c(num_customers, num_articles))
# dimnames(ui_mat) = list(cid_list, aid_list)
# dim(ui_mat)
# 
# # Keep only cases where we have observed the user
# ui_mat_sub = ui_mat[rowSums(ui_mat) > 0,]
# dim(ui_mat_sub)
# 
# # Keep only articles with some level of popularity
# articles_dy = readRDS("data/t8/articles_dy.rds")
# setDT(articles_dy)
# articles_dy = articles_dy[wk_predict == wks_test[i]][popularity > 0]
# 
# art_lsa_2 = art_lsa[sort(articles_dy$aid),]
# ui_mat_sub_2 = ui_mat_sub[, sort(articles_dy$aid)]
# 
# customer_profile = ui_mat_sub_2 %*% art_lsa_2
# customer_profile = as.matrix(customer_profile)
# dim(customer_profile)
# 
# topx = vector("list", nrow(customer_profile))
# topx_scores = vector("list", nrow(customer_profile))
# art_lsa_t = t(art_lsa_2)
# 
# res = Rcpp_crossprod(customer_profile, art_lsa_t, topx = 50);
# 
# # pb <- txtProgressBar(min = 0, max = nrow(customer_profile), style = 3)
# # for (i in seq_len(nrow(customer_profile))){
# #   scores = customer_profile[i,] %*% art_lsa_t
# #   topx[[i]] = order(-scores)[1:100]
# #   topx_scores[[i]] = scores[topx[[i]]]
# #   setTxtProgressBar(pb, i)
# # }
# 
# cbf = data.table(cid = as.numeric(rownames(customer_profile)), 
#                  preds = asplit(res[,1:50], 1),
#                  scores = asplit(res[,51:100], 1)
# )
# 
# cbf[, preds := lapply(preds, function(x) sort(articles_dy$aid)[x])]
# 
# saveRDS(cbf, "data/t8/cbf38.rds")
