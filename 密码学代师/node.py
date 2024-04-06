# 此程序实现对指定文件的解密操作
file = open('secret .daz','r')
content = file.read()
file.close()
my_list = content.split('X')
S = ''
length = 0
for element in my_list:
    if element != '':
        word = int(element, 16) # 转化为10进制才能使用chr转换为对应的字符。
        S += chr(word) 
print(S)
for char in S:
    if ord(char) != 32 and ord(char) != 9 and ord(char) != 10: # 通过空格、制表符、换行符对应的ASCII码来统计出现次数。
        length += 1
S1 = '<解密人>邓博文<情报总字数>' + str(length)
print(S1)
file = open('interpretation.txt','w')
file.write(S)
file.write('\n')
file.write(S1)
file.flush()
file.close()