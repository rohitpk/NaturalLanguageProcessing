

'''
https://www.kaggle.com/geraldm/headlines-for-major-crypto-from-2012-until-today/version/1#
https://toolbox.google.com/datasetsearch/search?query=A%20Million%20News%20Headlines&docid=ovMkg1bDln4S68FkAAAAAA%3D%3D
https://www.kaggle.com/therohk/india-news-publishing-trends-and-cities/data
'''

df = pd.read_csv('./abcnews-date-text.csv')
df['n_date'] =  pd.to_datetime(df['publish_date'].apply(str),format='%Y%m%d', errors='ignore')
df = df.set_index('n_date')    
one_year_data = df.loc['2016-01-01':'2017-01-01']
one_year_sample = one_year_data.sample(5010)
one_year_sample = one_year_sample.drop_duplicates('headline_text')
one_year_sample['class_tag'] = 0
one_year_sample.to_csv('random_one_year_news_5008.csv')




df1 = pd.read_csv('./Headline_Crypto.csv',encoding='ISO-8859-1')
sample = df1.sample(6000)
sample = sample.drop_duplicates('Headline')
sample.rename(columns={'Headline':"headline_text"},inplace=True)
sample['class_tag'] = 1
sample.to_csv('crypto_headline_5525.csv')



random_news = pd.read_csv('random_one_year_news_5008.csv')
crypto_data = pd.read_csv('crypto_headline_5525.csv')
random_news = random_news.sample(5000)
crypto_data = crypto_data.sample(5000)
final_data = crypto_data.append(random_news,ignore_index=True,sort=False)
final_data = final_data[['headline_text','class_tag']]



