# NLP#4 文本挖掘与词频统计

这一节里面我们轻松一下~来看一个好玩一点的动手项目。

在分析一段文本中，我们经常会用到一个常用的技巧，即分析某一个特定的词语出现的频率，或提取文本中出现频率较高的几个关键词。这个技巧就是我们今天要介绍的**词频统计（TFIDF）**。

一个词语出现的频率高低一定程度上体现了一个词语在这段文本中的“地位”。一个简单的例子就是小说，如果一个人物的名字在书中只出现过1次，那么我们基本上已经可以断定，这位老兄基本上就是一个不知名的吃瓜群众了；而如果一个人名反复出现，那么在研究这部作品的时候一定不能错过这个人物。

![Meme](C:\Users\tony\Desktop\RDFZ\AI\NLP\NLP#4\Meme.JPG)

同样的，词频统计同样活跃在各种各样的赋权问题上。研究表明，现如今超过83%的线上推荐系统用的都是词频统计技术。^[1]^ 

了解完了词频统计的用处，我们再来看一下它是怎么实现的吧。我们以《三国演义》为例，研究其中人物出场的次数的多少。

第一步先下载一份txt格式的《三国演义》，作为研究的对象。

然后在python中写好代码，用python中的jieba库来实现词频统计的功能。这里提供一份可以供读者参考的模板代码：

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:00:45 2018
reference: https://www.cnblogs.com/A9kl/p/9311246.html 
@author: rdfz-411
"""
# -*- encoding:utf-8 -*-

import jieba

content = open('三国演义.txt', 'r',encoding='utf-8').read()
words =jieba.lcut(content)#分词
excludes={"将军","却说","二人","后主","上马","不知","天子","大叫","众将","不可","主公","蜀兵","只见","如何","商议","都督","一人","汉中","不敢","人马","陛下","魏兵","天下","今日","左右","东吴","于是","荆州","不能","如此","大喜","引兵","次日","军士","军马"}#排除的词汇
words=jieba.lcut(content)
counts={}

for word in words:
    if len(word) == 1: # 排除单个字符的分词结果
        continue
    elif word == '孔明' or word == '孔明曰':
       real_word = '孔明'
    elif word == '关公' or word == '云长':
       real_word = '关羽'
    elif word == '孟德' or word == '丞相':
       real_word = '曹操'
    elif word == '玄德' or word == '玄德曰':
       real_word = '刘备'
    else:
        real_word =word
        counts[word] = counts.get(word, 0) + 1
for word in excludes:
    del(counts[word])
items=list(counts.items())
items.sort(key=lambda x:x[1],reverse=True)
for i in range(10):
    word,count=items[i]
    print("{0:<10}{1:>5}".format(word,count))
```

那么到底《三国演义》中谁出场次数最多呢？我们来看一下运行结果：

![](C:\Users\tony\Desktop\RDFZ\AI\NLP\NLP#4\Three Kingdoms.JPG)

大家有没有猜到呢？当然，这个简单的程序还有这许多可改进的地方，比如“卧龙”、”诸葛亮“、”亮“等很多称呼都可以用来指代孔明，而我们的程序却把这些分在了不同的人身上。读者有兴趣可以自行研究如何改进~

接下来，我们来研究一个衍生而来的小问题。有了每个词语的词频，我们应该如何直观地反应这些词语之间的相对关系呢？在这里我们引入词云图技术，作为一个直观的数据可视化做法。

所谓词云图，顾名思义，就是用图的方式展示词语的常用程度。使用频率高的词就会占有更大的空间，更容易被人们所发现，而使用频率低的词语则只会占有一小部分空间。

在现实操作中，选择一段想要的文本以及一张模板背景图片，python中wordcloud库中的函数就能够生成按照模板图片的形状生成一副词云图辣。

同样的在这里我们给出python的样例代码，这次我们研究的对象是余华的代表作《活着》：

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:07:40 2018
@author: rdfz-411
"""
# -*- encoding:utf-8 -*-

import jieba.analyse
from os import path
from scipy.misc import imread
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import imageio
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

if __name__ == "__main__":

    mpl.rcParams['font.sans-serif'] = ['FangSong']
    #mpl.rcParams['axes.unicode_minus'] = False

    content = open("huozhe.txt","rb").read()  ###!!!! 改文件名

    # tags extraction based on TF-IDF algorithm
    tags = jieba.analyse.extract_tags(content, topK=100, withWeight=False)
    text =" ".join(tags)
    text = str(text)
    print(text)
    # read the mask
    d = path.dirname(__file__)
    trump_coloring = imageio.imread(path.join(d, "trump.png"))
    stopwords = STOPWORDS.copy()
    stopwords.add("家珍")
    wc = WordCloud(font_path='simsun.ttc',
            background_color="white", max_words=1000, mask=trump_coloring,
            max_font_size=400, stopwords=stopwords,random_state=50)

    # generate word cloud 
    wc.generate(text)

    # generate color from image
    image_colors = ImageColorGenerator(trump_coloring)

    plt.imshow(wc)
    plt.axis("off")
    plt.show()
```

最后的就能得到我们想要的成果啦~

![Cloud](C:\Users\tony\Desktop\RDFZ\AI\NLP\NLP#4\Cloud.jpeg)

（PS：大家猜一猜展示的词云图是以什么为模板生成出来的呢？( ´´ิ∀´ิ`））

[1] Breitinger, Corinna; Gipp, Bela; Langer, Stefan (2015-07-26). ["Research-paper recommender systems: a literature survey"](http://nbn-resolving.de/urn:nbn:de:bsz:352-0-311312) (Submitted manuscript). *International Journal on Digital Libraries*. **17** (4): 305–338. [doi](https://en.wikipedia.org/wiki/Digital_object_identifier):[10.1007/s00799-015-0156-0](https://doi.org/10.1007%2Fs00799-015-0156-0). [ISSN](https://en.wikipedia.org/wiki/International_Standard_Serial_Number) [1432-5012](https://www.worldcat.org/issn/1432-5012).