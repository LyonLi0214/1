import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP
import chardet
import re
from matplotlib.font_manager import FontProperties

# 检测文件编码
with open('1.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print(f"文件编码: {encoding}")

# 读取 CSV 文件
file_path = '1.csv'  # 替换为您的 CSV 文件路径
df = pd.read_csv(file_path, sep='\t', encoding=encoding)  # 使用检测到的编码

# 打印列名
print("列名:", df.columns.tolist())

# 查看数据的前几行
print("数据预览:")
print(df.head())

# 检查评论列是否存在
if '评论内容' in df.columns:
    # 数据清洗
    def clean_text(text):
        # 去除特殊字符和标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 去除数字
        text = re.sub(r'\d+', '', text)
        # 转换为小写
        text = text.lower()
        return text

    # 清洗评论内容
    df['评论内容'] = df['评论内容'].apply(clean_text)

    # 去除空评论
    df = df[df['评论内容'].str.strip() != '']

    # 进行情感分析
    def analyze_sentiment(text):
        s = SnowNLP(text)
        return s.sentiments

    df['情感得分'] = df['评论内容'].apply(analyze_sentiment)

    # 分类情感得分
    def classify_sentiment(score):
        if score >= 0.6:
            return '积极'
        elif score <= 0.4:
            return '消极'
        else:
            return '中立'

    df['情感分类'] = df['情感得分'].apply(classify_sentiment)

    # 统计各情感分类的数量
    sentiment_counts = df['情感分类'].value_counts()

    # 设置中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=12)

    # 创建条形图
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.xlabel('情感分类', fontproperties=font)
    plt.ylabel('数量', fontproperties=font)
    plt.title('评论情感分布', fontproperties=font)
    plt.xticks(rotation=0, fontproperties=font)
    plt.show()
else:
    print("警告: 数据中没有名为 '评论内容' 的列")