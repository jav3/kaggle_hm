cd("/Users/javidali/Downloads/Kaggle/hm")

using Arrow
using CSV
using DataFrames
using Dates
using Pipe

# Convert data to arrow format
Arrow.write("data/transactions.arrow", CSV.File("data/transactions_train.csv", types = Dict(:t_dat=>Date, :customer_id=>String, :article_id=>String, :price=>Float64, :sales_channel_id=>Int8)))
Arrow.write("data/customers.arrow", CSV.File("data/customers.csv", types = Dict(:customer_id=>String, :FN=>String, :Active=>String, :fashion_news_frequency=>String)))

Arrow.write("data/articles.arrow", CSV.File("data/articles.csv", types = Dict(:article_id=>String)))

# Simplify file sizes
customers = DataFrame(Arrow.Table("data/customers.arrow"))
customers_dict = Dict(val => idx for (idx, val) in enumerate(customers.
customer_id))
customer_mapping = DataFrame("key" => collect(keys(customers_dict)), "val" => collect(values(customers_dict)))

articles = DataFrame(Arrow.Table("data/articles.arrow"))
articles_dict = Dict(val => idx for (idx, val) in enumerate(articles.article_id))
article_mapping = DataFrame("key" => collect(keys(articles_dict)), "val" => collect(values(articles_dict)))

transactions[:, :aid] = getindex.(Ref(articles_dict), transactions.article_id)
select!(transactions, Not(:article_id))
transactions[:, :cid] = getindex.(Ref(customers_dict), transactions.customer_id)
select!(transactions, Not(:customer_id))

customers[:, :cid] = getindex.(Ref(customers_dict), customers.customer_id)
select!(customers, Not(:customer_id))

articles[:, :aid] = getindex.(Ref(articles_dict), articles.article_id)
select!(articles, Not(:article_id))

Arrow.write("data/transactions_m.arrow", transactions)
Arrow.write("data/customers_m.arrow", customers)
Arrow.write("data/articles_m.arrow", articles)
Arrow.write("data/customer_mapping.arrow", customer_mapping)
Arrow.write("data/article_mapping.arrow", article_mapping)