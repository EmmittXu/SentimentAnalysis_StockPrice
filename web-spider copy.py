
import os
from selenium import webdriver
import time




if __name__ == '__main__':
	url = 'http://www.reuters.com/article/us-apple-china-streaming-idUSKCN0RU07M20150930'
	driver = webdriver.PhantomJS()
	driver.get(url)
	news = ""
	file_name = 'news.txt'
	f = open(file_name,'w')
	news_list = driver.find_element_by_id('article-text')
	news_text_list = news_list.find_elements_by_tag_name('p')
	for li in news_text_list:
		news = news + li.text.encode('utf-8')
	f.write(news)





