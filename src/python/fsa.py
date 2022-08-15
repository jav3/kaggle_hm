# Setup
import pyarrow as pa
from fse import Vectors, Average, IndexedList
from nltk.corpus import stopwords
from fse.models import uSIF
from numpy import savetxt
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

articles = pa.ipc.open_file('./data/articles_m.arrow').read_pandas()
articles['detail_desc'] = articles['detail_desc'].astype(str)
articles['desc'] = articles['detail_desc'].apply(tokenizer.tokenize)
articles['desc'] = articles['desc'].apply(lambda words: [word.lower() for word in words if word not in stop])

vecs = Vectors.from_pretrained("glove-wiki-gigaword-100")

model = uSIF(vecs, workers=1, lang_freq="en")

desc = articles.desc.tolist()

model.train(IndexedList(desc))

savetxt('data/t8/fsa_usif.csv', model.sv.vectors, delimiter=',')



