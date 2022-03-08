import csv

file = open('data//test.csv' , 'a' , encoding='utf-8')

csv_writer = csv.writer(file)
csv_writer.writerow(['姓名'  , '性别' , '年龄'])
csv_writer.writerow(['张三' , [1,2,3,4,5] , '3'])