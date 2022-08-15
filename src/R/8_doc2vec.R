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
library(text2vec)
library(quanteda.textstats)
library(word2vec)
library(doc2vec)

setwd("~/Downloads/Kaggle/hm")


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

# Similar articles --------------------------------------------------------

art_txt = articles_data[, .(doc_id = aid, text = paste(prod_name, product_type_name, product_group_name, 
                                                       graphical_appearance_name, colour_group_name, perceived_colour_value_name,
                                                       perceived_colour_master_name, department_name, index_name, index_group_name,
                                                       section_name, garment_group_name, detail_desc, sep = ". "))]
art_txt[, text := txt_clean_word2vec(text)]


d2v = paragraph2vec(art_txt, type = "PV-DBOW", dim = 50,lr = 0.05, iter = 10,
                        window = 15, hs = TRUE, negative = 0,
                        sample = 0.00001, min_count = 5, threads = 4)

sentences = strsplit(setNames(art_txt$text, art_txt$doc_id), split = " ")
nn = predict(d2v, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 50)

nn_2 = rbindlist(nn)
setnames(nn_2, 1:3, c("aid", "linked_aid", "score"))
nn_2[, aid := as.numeric(aid)]
nn_2[, linked_aid := as.numeric(linked_aid)]

if (F){
  saveRDS(nn_2, "data/t8/articles_doc2vec.rds")
}

# art_corpus = corpus(articles_data[, paste(prod_name, product_type_name, product_group_name, 
#                                           graphical_appearance_name, colour_group_name, perceived_colour_value_name, 
#                                           perceived_colour_master_name, department_name, index_name, index_group_name, 
#                                           section_name, garment_group_name, detail_desc, sep = ". ")])
# docvars(art_corpus, "aid") = articles_data$aid
# 
# art_tokens = tokens(art_corpus, remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T)
# 
# art_dfm = dfm(art_tokens)
# art_dfm = dfm_tolower(art_dfm)
# art_dfm = dfm_remove(art_dfm, stopwords("en"))
# art_dfm = dfm_wordstem(art_dfm)
# art_dfm = dfm_trim(art_dfm, min_docfreq = 1)
# art_dfm = dfm_tfidf(art_dfm)
# dim(art_dfm)
# 
# art_toks = tokens_select(art_tokens, art_dfm %>% featnames(), padding = TRUE)
# 
# art_fcm = fcm(art_toks, context = "window", count = "weighted", weights = 1 / (1:5), tri = TRUE)
# 
# 
# # Fit model ---------------------------------------------------------------
# 
# glove = GlobalVectors$new(rank = 50, x_max = 10)
# wv_main = glove$fit_transform(art_fcm, n_iter = 100, convergence_tol = 0.01, n_threads = 8)
# wv_context = glove$components
# word_vectors = wv_main + t(wv_context)
# 
# if (0){
#   a = word_vectors["Ladieswear", , drop = FALSE] -
#     word_vectors["bra", , drop = FALSE] +
#     word_vectors["Premium", , drop = FALSE]
#   cos_sim <- textstat_simil(x = as.dfm(word_vectors), y = as.dfm(a),
#                             method = "cosine")
#   head(sort(cos_sim[, 1], decreasing = TRUE), 5)
# }
# 
# art_glove = as.dfm(word_vectors)
# 
# # Article similarity computation
# batch_split = (seq(nrow(art_glove)) %/% 2000) + 1
# res = vector(mode = "list", length(unique(batch_split)))
# pb <- txtProgressBar(min = 0, max = length(res), style = 3)
# for (i in unique(batch_split)){
#   setTxtProgressBar(pb, i)
#   res[[i]] = textstat_simil(art_glove, art_glove[which(batch_split == i),], margin = "documents", method = "cosine", min_simil = 0.5)
# }
# 
# res_2 = unlist(lapply(res, as.list), recursive = F)
# res_2 = Map(function(x, nm) data.table(aid = nm, linked_aid = 
#                                          as.numeric(gsub("text", "", names(x))), score = x), 
#             res_2, 
#             as.numeric(gsub("text", "", names(res_2))))
# 
# res_2 = rbindlist(res_2, use.names = T, fill = T)
# 
# if (F){
#   saveRDS(res_2, "data/t8/articles_glove.rds")
# }

