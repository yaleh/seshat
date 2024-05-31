import chardet

# 检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        if result['encoding']=='ISO-8859-1':
            return 'gbk'
        else:
            return result['encoding']