library(glmnet)
library(Matrix)

trn = curr_purchases %>% merge(articles_dy[yr == curr_yr & wk == curr_wk - 1], by = "aid", all.x = T, all.y = T) %>% 
  merge(customers_dy[yr == curr_yr & wk == curr_wk - 1], by = "cid", all.x = T, all.y = F) 
trn[is.na(purchased), purchased := 0]

trn = trn[purchased == 1][, aid := as.factor(aid)]

trn[is.na(popularity), popularity := 0]
trn[is.na(last_w_popularity), last_w_popularity := 0]
trn[, not_on_market := is.na(days_on_market)]
trn[is.na(days_on_market), days_on_market := 0]
trn[is.na(new_arrival_1m), new_arrival_1m := F]

trn[, aid_ct := .N, by = .(aid)]
trn = trn[aid_ct > 300]

trn_mat = sparse.model.matrix(~ popularity + last_w_popularity + not_on_market + days_on_market + new_arrival_1m - 1, data = trn)
trn_y = as.factor(as.character(trn$aid)) # model.matrix(~ y - 1, data = data.frame(y = as.factor(as.character(trn$aid))))

lambdas = exp(seq(0,-10,-0.5))
fit = cv.glmnet(x = trn_mat, y = trn_y, family = "multinomial", type.multinomial = "grouped", 
                lambda = lambdas, nfolds = 3)
plot(fit)

predict(fit, newx = trn_mat[1,,drop = F], type = "response") %>% {which.max(.)}
