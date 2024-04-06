# 此函数显示疫苗下一针接种情况，并且可以实现查询多人接种情况。
from datetime import date,timedelta
def date_now():   # date_now函数询问今天日期，以及需要查询的人数。
    today = date.fromisoformat(input('请输入今天的日期(example:2024-03-08):'))
    person_number = int(input('请问你要查询几个人：'))
    return today,person_number
today,person_number = date_now()
def date_input():  # date_input函数询问已经接种针数以及最近一次接种日期并将其分别作为key和value组成一个字典，添加到数组中。
    store_list = []
    for _ in range(person_number) : # i在后续没有使用，用_代替i占位
        injection = int(input('请输入已经接种了几针：'))
        near_date = date.fromisoformat(input('请输最近一次的接种日期(example:2024-03-08):'))
        store_list.append({injection:near_date})
    return store_list
data_list = date_input()
def date_output():   # date_output函数计算下一针课接种的时间和是否已达到接种时间并返回一个包含这些数据的列表。
    new_list = []
    for i in range(person_number):
        dict_element = data_list[i]
        for key,value in dict_element.items():
            if key == 0:
                new_list.append({True:str(today)}) # 为正常显示日期，将date转为string。
            if key == 1:
                next_date = value + timedelta(days=30)
                new_list.append({(next_date < today):str(next_date)})
            if key == 2:
                next_date = value + timedelta(days=180)
                new_list.append({(next_date < today):str(next_date)})
            if key == 3:
                new_list.append({False:''})
    return new_list
result_list = date_output() 
print(result_list)