# encoding=utf-8
import jieba

jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["昨天今天明天一起玩耍廢上午中午下午晚上床吃飯飯睡覺覺抱抱枕頭摸摸頭微笑容忍耐性慾望希望遠方向東西南北中心愛心情好心情不好不好"]
for str in strs:
    seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
    print("Paddle Mode: " + '/'.join(list(seg_list)))
print()

seg_list = jieba.cut("昨天今天明天一起玩耍廢上午中午下午晚上床吃飯飯睡覺覺抱抱枕頭摸摸頭微笑容忍耐性慾望希望遠方向東西南北中心愛心情好心情不好不好", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式
print()

seg_list = jieba.cut("昨天今天明天一起玩耍廢上午中午下午晚上床吃飯飯睡覺覺抱抱枕頭摸摸頭微笑容忍耐性慾望希望遠方向東西南北中心愛心情好心情不好不好", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
print()

seg_list = jieba.cut("昨天今天明天一起玩耍廢上午中午下午晚上床吃飯飯睡覺覺抱抱枕頭摸摸頭微笑容忍耐性慾望希望遠方向東西南北中心愛心情好心情不好不好")  # 默认是精确模式
print(", ".join(seg_list))
print()

seg_list = jieba.cut_for_search("昨天今天明天一起玩耍廢上午中午下午晚上床吃飯飯睡覺覺抱抱枕頭摸摸頭微笑容忍耐性慾望希望遠方向東西南北中心愛心情好心情不好不好")  # 搜索引擎模式
print(", ".join(seg_list))
print()

########################################################################################################################33

import jieba.posseg as pseg

text = '不管他不受啥影響還是不會被指定還是怎樣,只要他能夠拿來L就是可以拿來L ( 感覺很廢話 ),反之像是淡島之類不能拿來L的,就不能用冥神吃掉'
words = pseg.cut(text)
print([word for word in words])
