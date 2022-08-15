# Working dir
cd("/Users/javidali/Downloads/Kaggle/hm")

# Packages
using CSV
using DataFrames
using Dates
using Pipe
using Query
using Arrow
using Distributed
using ProgressMeter
using Statistics
using SparseArrays
using Flux
using LinearAlgebra
using TextAnalysis
using TextModels
using Embeddings
using StatsBase
include("src/julia/helpers.jl")
using Lathe
using Random

# Import data
transactions = DataFrame(Arrow.Table("data/transactions_m.arrow"))
customers = DataFrame(Arrow.Table("data/customers_m.arrow"))
articles = DataFrame(Arrow.Table("data/articles_m.arrow"))

num_articles = size(articles, 1)
product_group_name = spzeros(num_articles, num_articles)



@showprogress for i in 1:num_articles
    product_group_name[articles.product_group_name.refs .== articles.product_group_name.refs[i], i] .= 1
end

product_group_name = [findall(x -> x .== val, articles.product_group_name.refs) for val in articles.product_group_name.refs]

map(Iterators.product(articles.product_group_name, articles.product_group_name)) do (x,y)
    x == y ? 1 : 0
end