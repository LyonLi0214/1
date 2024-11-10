
pip install textblob


import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import chardet

# 检测文件编码
with open('C:/Users/Lyon/Desktop/result/1Z1DNY5Egc.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print(f"文件编码: {encoding}")

# 读取 CSV 文件
file_path = 'C:/Users/Lyon/Desktop/result/1Z1DNY5Egc.csv'  # 替换为您的 CSV 文件路径
df = pd.read_csv(file_path, sep='\t', encoding=encoding)  # 使用检测到的编码

# 打印列名
print("列名:", df.columns.tolist())

# 查看数据的前几行
print("数据预览:")
print(df.head())

# 检查评论列是否存在
if '评论内容' in df.columns:
    # 初始化情感分析器
    analyzer = SentimentIntensityAnalyzer()

    # 进行情感分析
    def analyze_sentiment(text):
        vs = analyzer.polarity_scores(text)
        return vs['compound']

    df['情感得分'] = df['评论内容'].apply(analyze_sentiment)

    # 分类情感得分
    def classify_sentiment(score):
        if score >= 0.05:
            return '积极'
        elif score <= -0.05:
            return '消极'
        else:
            return '中立'

    df['情感分类'] = df['情感得分'].apply(classify_sentiment)

    # 统计各情感分类的数量
    sentiment_counts = df['情感分类'].value_counts()

    # 创建条形图
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.xlabel('情感分类')
    plt.ylabel('数量')
    plt.title('评论情感分布')
    plt.xticks(rotation=0)
    plt.show()
else:
    print("警告: 数据中没有名为 '评论内容' 的列")