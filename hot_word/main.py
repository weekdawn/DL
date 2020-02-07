import jieba

words = "中华人民共和国123"

print(jieba.lcut(words, cut_all=True))
