import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import json


proxies = {'http' : 'http://' + '112.31.16.81:7890'} #* 整个代理，不行就换了
#proxies = proxies


def getfilminfo(url,headers):
    r = requests.get(url, proxies = proxies , headers=headers, timeout=10)
    r.raise_for_status()
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')
    # 片名
    name = soup.find(attrs={'property': 'v:itemreviewed'}).text
    # 上映年份
    yearIterator = soup.find_all(attrs={'property': 'v:initialReleaseDate'})
    year = []
    for i in yearIterator:
        year.append(i.text)
    
    #*year = soup.find(attrs={'class': 'year'}).text#.replace('(','').replace(')','')
    
    # 评分
    score = soup.find(attrs={'property': 'v:average'}).text
    # 评价人数
    votes = soup.find(attrs={'property': 'v:votes'}).text
    
    infos = soup.find(attrs={'id': 'info'}).text.split('\n')[1:11]
    # 导演
    director = infos[0].split(': ')[1]
    # 编剧
    scriptwriter = infos[1].split(': ')[1].split(' / ')
    # 主演
    actor = infos[2].split(': ')[1].split(' / ')
    # 类型
    filmtype = infos[3].split(': ')[1].split(' / ')
    # 国家/地区
    area = infos[4].split(': ')[1]
    if '.' in area:
        area = infos[5].split(': ')[1].split(' / ')
        # 语言
        language = infos[6].split(': ')[1].split(' / ')
        # 片长
        time = infos[8].split(': ')[1].split(' / ')
    else:
        area = infos[4].split(': ')[1].split(' / ')
        # 语言
        language = infos[5].split(': ')[1].split(' / ')
        # 片长
        time = infos[7].split(': ')[1].split(' / ')

    
    filminfo = {
        "片名" : name , 
        "导演" : director , 
        "编剧" : scriptwriter , 
        "主演" : actor , 
        "类型" : filmtype , 
        "制片国家/地区" : area , 
        "语言" : language , 
        "上映日期" : year , 
        "片长" : time , 
        "评分" : score , 
        "评价人数" : votes
    }
    return filminfo

def getonepagelist(url, headers , filename):
    try:
        r = requests.get(url, proxies = proxies , headers=headers, timeout=10)
        r.raise_for_status()
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'html.parser')
        lists = soup.find_all(attrs={'class': 'hd'})
        for index , lst in enumerate(lists):
            
            href = lst.a['href']
            
            print(index+1)
            
            time.sleep(np.random.uniform(5,8)) #* 睡一会儿
            
            
            json.dump(getfilminfo(href, headers) , filename , ensure_ascii=False , indent = 4) #* 写入到数据当中
    except:
            print('getonepagelist error!')

if __name__ == '__main__':
    head = {  # 模拟浏览器头部信息，向豆瓣服务器发送消息
            "User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 112.31.16.81  Safari / 537.36"
        }
    JsonFile = open("data\\OriginalData_2.json" , "a" , encoding="utf-8")
    
    
    for i in range(9,10):
        print(f'正在爬取第{i+1}页,请稍等...')
        url = 'https://movie.douban.com/top250?start={}&filter='.format(i * 25)    
        getonepagelist(url , head , JsonFile)

    
    JsonFile.close()
    
    '''
    注意，因为写入json时由于豆瓣的反爬，我写入了250个字典，但是json.load的解码过程不允许解码多个字典，所以需要用vscode自带的ctrl+f进行替换，将这250个字典都塞入到一个数组当中进行读取，这也是为什么我需要backup的缘故
    '''
    
    
# s = etree.HTML(test.text)

# fileOb = open('Sec_exp\\data\\bawangbieji.html','w',encoding='utf-8')     #打开一个文件，没有就新建一个
# fileOb.write(test.text)











