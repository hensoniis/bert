# bert_ner
中文實體辨識


#### 中文 Named Entity Recognition 命名實體識別 - NER

> 命名實體識別（英語：Named Entity Recognition，簡稱NER），又稱作專名識別、命名實體，是指識別文本中具有特定意義的實體，主要包括人名、地名、機構名、專有名詞等，以及時間、數量、貨幣、比例數值等文字。指的是可以用專有名詞（名稱）標識的事物，一個命名實體一般代表唯一一個具體事物個體，包括人名、地名等。 -- wiki

##### 來源：[ProHiryu/bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner "ProHiryu/bert-chinese-ner")

基本上就是可以抓出一些特定物件，人名、地名、機構名、專有名詞，之前也有抓過金額、日期、時間、金流渠道等。 這次的資料集則是只有人名、地名、機構名。

為求方便，這次我就改成使用 [Kashgari](https://github.com/BrikerMan/Kashgari "Kashgari") 提供的ChineseDailyNerCorpus，安裝方式請看[官網](https://kashgari-zh.bmio.net/#_3 "官網")
或是透過pip
```
pip install kashgari
```

------------

# 快速啟動
``` python3 bert_ner.py ```

# 訓練部分
### 訓練參數
這是Kashgari官方給的

```
batch_size: 2317
sequence_length: 100
epochs: 200
```

個人使用

```
train_batch_size = 16
eval_batch_size = 16
predict_batch_size = 16
learning_rate = 5e-5
num_train_epochs = 5
```

但由於我沒有機器跑那麼久，這次提供的結果我只跑了很少的部分


### 輸入資料

基本上規則滿簡單的
1. 不需特別標註的字（非訓練目標），則標註成 **O**
2. 需標註的字，首字使用 **Ｂ** 開頭，其餘自使用**Ｉ**開頭
3. 需標註的字，首字之後放標籤代號例如 組織名 -> **B-ORG** or **I-ORG** 其中一個

範例如下

| 內容  | 標註  |
| ------------ | ------------ |
|中|B-ORG
|共|I-ORG
|中|I-ORG
|央|I-ORG
|致|O
|中|B-ORG
|国|I-ORG
|致|I-ORG
|公|I-ORG
|党|I-ORG
|十|I-ORG
|一|I-ORG
|大|I-ORG
|的|O
|贺|O
|词|O
| |
|各|O
|位|O
|代|O
|表|O
|、|O
|各|O
|位|O
|同|O
|志|O
|：|O
||
|在|O
|中|B-ORG
|国|I-ORG
|致|I-ORG
|公|I-ORG
|党|I-ORG
|第|I-ORG
|十|I-ORG
|一|I-ORG
|次|I-ORG
|全|I-ORG
|国|I-ORG
|代|I-ORG
|表|I-ORG
|大|I-ORG
|会|I-ORG
|隆|O
|重|O
|召|O
|开|O
|之|O
|际|O
|，|O
|中|B-ORG
|国|I-ORG
|共|I-ORG
|产|I-ORG
|党|I-ORG
|中|I-ORG
|央|I-ORG
|委|I-ORG
|员|I-ORG
|会|I-ORG
|谨|O
|向|O
|大|O
|会|O
|表|O
|示|O
|热|O
|烈|O
|的|O
|祝|O
|贺|O
|，|O
|向|O
|致|B-ORG
|公|I-ORG
|党|I-ORG
|的|O
|同|O
|志|O
|们|O


查看一下 ChineseDailyNerCorpus 資料

```
from kashgari.corpus import ChineseDailyNerCorpus

train_x, train_y = ChineseDailyNerCorpus.load_data('train')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
test_x, test_y  = ChineseDailyNerCorpus.load_data('test')

def load_data2set(set_x, set_y, max_num = 50000):
      data_set = []
      idx = 0
      sent_text,sent_slot="",""
      for label,ans in zip(set_x,set_y):
          idx+=1
          if idx > max_num and max_num !=-1:
            break
          sent_text = " ".join(label)
          sent_slot = " ".join(ans)
          data_set.append([sent_text,sent_slot])

      print("data_set: ",data_set[:50])
      return data_set
print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(valid_x)}")
print(f"test data count: {len(test_x)}")


load_data2set(train_x[:10],train_y[:10])`
```
result:
```
train data count: 20864
validate data count: 2318
test data count: 4636
data_set:  [['这 是 古 代 乱 世 中 的 白 日 梦 ： 在 山 间 清 流 中 荡 一 叶 扁 舟 独 行 ， 穿 过 一 个 幽 暗 的 洞 口 ， 眼 前 豁 然 开 朗 ： 桃 花 林 夹 岸 ， 芳 华 鲜 美 ， 落 英 缤 纷 。', 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['大 街 上 摩 肩 接 踵 ， 人 们 扶 老 携 幼 ， 举 家 出 动 。', 'O O O O O O O O O O O O O O O O O O O O'], ['换 言 之 ， 理 解 与 划 分 新 时 期 小 说 流 派 ， 理 论 要 落 实 到 实 践 的 生 命 实 体 中 ， 而 非 拿 实 践 去 迎 合 理 论 框 框 。', 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['受 外 界 因 素 的 影 响 ， 今 天 菲 律 宾 比 索 对 美 元 的 汇 率 再 次 突 破 4 0 ∶ 1 防 线 。', 'O O O O O O O O O O O B-LOC I-LOC I-LOC O O O O O O O O O O O O O O O O O O O'], ['钟 添 发 同 时 还 宣 布 了 其 他 9 个 项 目 的 亚 运 会 目 标 。', 'B-PER I-PER I-PER O O O O O O O O O O O O O B-LOC O O O O O'], ['而 我 们 在 很 长 一 个 时 期 内 ， 却 用 一 种 宗 派 主 义 的 态 度 来 对 待 外 国 史 学 ， 这 就 使 我 们 的 史 学 故 步 自 封 ， 知 识 结 构 老 化 。', 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['这 只 穿 兜 肚 、 哼 小 曲 、 舞 龙 头 的 鸭 子 淘 气 又 可 爱 ， 能 用 一 支 神 笔 对 付 繁 重 的 功 课 ， 用 橡 皮 泥 吹 出 “ 孩 子 乐 园 ” 。', 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['哥 伦 比 亚 大 学 的 研 究 人 员 通 过 对 这 个 家 族 进 行 脱 氧 核 糖 核 酸 检 测 和 利 用 实 验 鼠 进 行 实 验 ， 发 现 在 第 八 个 染 色 体 的 某 个 区 域 存 在 一 种 脱 毛 基 因 。', 'B-ORG I-ORG I-ORG I-ORG I-ORG I-ORG O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['他 们 这 样 做 ， 为 某 些 人 私 设 “ 小 金 库 ” 开 了 方 便 之 门 ， 原 本 该 上 缴 国 家 的 税 款 有 可 能 装 进 个 人 的 腰 包 。', 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'], ['而 具 有 创 新 头 脑 的 人 才 要 靠 教 育 来 培 养 。', 'O O O O O O O O O O O O O O O O O O']]
[['这 是 古 代 乱 世 中 的 白 日 梦 ： 在 山 间 清 流 中 荡 一 叶 扁 舟 独 行 ， 穿 过 一 个 幽 暗 的 洞 口 ， 眼 前 豁 然 开 朗 ： 桃 花 林 夹 岸 ， 芳 华 鲜 美 ， 落 英 缤 纷 。',
  'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['大 街 上 摩 肩 接 踵 ， 人 们 扶 老 携 幼 ， 举 家 出 动 。',
  'O O O O O O O O O O O O O O O O O O O O'],
 ['换 言 之 ， 理 解 与 划 分 新 时 期 小 说 流 派 ， 理 论 要 落 实 到 实 践 的 生 命 实 体 中 ， 而 非 拿 实 践 去 迎 合 理 论 框 框 。',
  'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['受 外 界 因 素 的 影 响 ， 今 天 菲 律 宾 比 索 对 美 元 的 汇 率 再 次 突 破 4 0 ∶ 1 防 线 。',
  'O O O O O O O O O O O B-LOC I-LOC I-LOC O O O O O O O O O O O O O O O O O O O'],
 ['钟 添 发 同 时 还 宣 布 了 其 他 9 个 项 目 的 亚 运 会 目 标 。',
  'B-PER I-PER I-PER O O O O O O O O O O O O O B-LOC O O O O O'],
 ['而 我 们 在 很 长 一 个 时 期 内 ， 却 用 一 种 宗 派 主 义 的 态 度 来 对 待 外 国 史 学 ， 这 就 使 我 们 的 史 学 故 步 自 封 ， 知 识 结 构 老 化 。',
  'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['这 只 穿 兜 肚 、 哼 小 曲 、 舞 龙 头 的 鸭 子 淘 气 又 可 爱 ， 能 用 一 支 神 笔 对 付 繁 重 的 功 课 ， 用 橡 皮 泥 吹 出 “ 孩 子 乐 园 ” 。',
  'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['哥 伦 比 亚 大 学 的 研 究 人 员 通 过 对 这 个 家 族 进 行 脱 氧 核 糖 核 酸 检 测 和 利 用 实 验 鼠 进 行 实 验 ， 发 现 在 第 八 个 染 色 体 的 某 个 区 域 存 在 一 种 脱 毛 基 因 。',
  'B-ORG I-ORG I-ORG I-ORG I-ORG I-ORG O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['他 们 这 样 做 ， 为 某 些 人 私 设 “ 小 金 库 ” 开 了 方 便 之 门 ， 原 本 该 上 缴 国 家 的 税 款 有 可 能 装 进 个 人 的 腰 包 。',
  'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O'],
 ['而 具 有 创 新 头 脑 的 人 才 要 靠 教 育 来 培 养 。',
  'O O O O O O O O O O O O O O O O O O']]
```


### 評估

```
eval_f = 0.7191007
eval_precision = 0.5988906
eval_recall = 0.9756849
global_step = 13040
loss = 20.316486
```
------------
# 預測

有輸出一份結果 在 [bert_ner/output/test_samples_results.xlsx](https://github.com/Chunshan-Theta/bert_ner/blob/master/output/test_samples_results.xlsx)裡面

| 模型預測  | 人工答案  | 內容 |
| ------------ | ------------ | ------------ |
|[CLS] | [CLS] | [CLS] |
|O|O|学
|O|O|校
|O|O|还
|O|O|组
|O|O|织
|O|O|特
|O|O|困
|O|O|生
|O|O|参
|O|O|加
|O|O|丰
|O|O|富
|O|O|多
|O|O|彩
|O|O|的
|O|O|社
|O|O|会
|O|O|活
|O|O|动
|O|O|，
|O|O|登
|B-LOC|B-LOC|天
|I-LOC|I-LOC|安
|I-LOC|I-LOC|门
|O|O|、
|O|O|听
|O|O|音
|O|O|乐
|O|O|会
|O|O|等
|O|O|，
|O|O|让
|O|O|孩
|O|O|子
|O|O|们
|O|O|感
|O|O|受
|O|O|到
|O|O|集
|O|O|体
|O|O|的
|O|O|温
|O|O|暖
|O|O|。
|[SEP]|[SEP]|[SEP]
|[CLS]|[CLS]|[CLS]
|O|O|所
|O|O|谓
|O|O|批
|O|O|判
|O|O|继
|O|O|承
|O|O|，
|O|O|当
|O|O|然
|O|O|是
|O|O|要
|O|O|有
|O|O|所
|O|O|批
|O|O|判
|O|O|，
|O|O|但
|O|O|主
|O|O|要
|O|O|还
|O|O|是
|O|O|要
|O|O|把
|O|O|优
|O|O|秀
|O|O|的
|O|O|传
|O|O|统
|O|O|继
|O|O|承
|O|O|下
|O|O|来
|O|O|，
|O|O|发
|O|O|扬
|O|O|光
|O|O|大
|O|O|。
|[SEP]|[SEP]|[SEP]
|[CLS]|[CLS]|[CLS]

# 額外
去抓[這邊的資料](https://github.com/ProHiryu/bert-chinese-ner/tree/master/data)
改檔案名之後放到`data/`資料夾後也跑跑看，可能因為資料的關係，評估的效果更好

```
eval_f = 0.9072733
eval_precision = 0.85459185
eval_recall = 0.9898753
global_step = 23745
loss = 6.551677

```
