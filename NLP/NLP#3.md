## NLP#3 预处理

这一节里面我们主要来学习文本预处理。

首先我们先从一个例子开始，考虑下面一段文本：

```c++
早上几年的时候，家珍还是一个女学生。那时候城里有夜校了，家珍穿着月白色的旗袍，提着一盏小煤油灯，和几个女伴去上学。我是在拐弯处看到她，她一扭一扭地走过来，高跟鞋敲在石板路上，滴滴答答像是在下雨，我眼睛都看得不会动了，家珍那时候长得可真漂亮，头发齐齐地挂到耳根，走去时旗袍在腰上一皱一皱，我当时就在心里想，我要她做我的女人。
																		-余华 《活着》
```

如果要读者翻译这段文本的话，大家应该怎么处理呢？会有哪些部分比较困难呢？

我们不妨看一看Google Translate是怎么处理这段文本的：

```c++
In the mornings, Jia Zhen was still a female student. At that time, there was a night school in the city. Jiazhen wore a white cheongsam, carrying a small kerosene lamp, and went to school with several female companions. I saw her at the corner. She walked over and twisted. The high-heeled shoes hit the stone road. The drip was like it was raining. My eyes didn’t move. Jiazhen was so real. Beautiful, the hair hangs in the roots of the ear. When I walk, the cheongsam wrinkles around the waist. I thought about it at the time, I want her to be my woman.
```

lǎi lǎi lǎi，我们开始研（tǔ）究（cáo）一下。首先，“早上几年”被处理成了"in the mornings"，很明显只看到了“早上”，而忽视了“早上几年”这样的整体。第二，“一扭一扭”这样以意会为主的文本被翻译为“twisted”，较为突兀又没有体现出原文想要表达的场景。再其次，作为文学作品的一个片段，翻译后的文本已经失去了原有语言中的美感，变得较为僵硬，拗口。

不难想到，在翻译的时候，需要对文意进行较深入的了解，否则就会出现上面的满满槽点。如需要了解文意，那么要做的工作就是文本预处理。文本预处理包括以下内容：**分词**，**词义消歧**，**词性标注**，**命名实体识别**，**依存句法分析**，**语义角色标注**。这一节里面我们主要关注命名实体识别和词性标注两部分。

### 命名实体识别与分类

回想上一节中我们提到的文本：

```c++
国务院总理李克强调研上海外高桥
```

我们在上一节中提出，对于机器来说，知道要在“强”和“调”之间断开一下，而不是把“强调”分成一个词是有一定困难的。在这里我们引出“命名实体”的概念，先来看一看定义：

```c++
In information extraction, a named entity is a real-world object, such as persons, locations, organizations, products, etc., that can be denoted with a proper name. It can be abstract or have a physical existence. Examples of named entities include Barack Obama, New York City, Volkswagen Golf, or anything else that can be named. Named entities can simply be viewed as entity instances.
    																		-Wikipedia
```

即命名实体指现实生活中的有特定含义的人名、地名、机构名称、专有名词。文本中的李克强总理就是这样的一个有特定含义的人名，因此是一个命名实体。命名实体如果不加以特定识别，容易被分拆到邻近的词语中，造成语意混乱。因此，**命名实体识别与分类**（**Named Entity Recognition and Classification, NERC**）可以说是文本预处理中非常重要的一步。

命名实体识别应该包含哪些部分呢？首先我们需要确定一个命名实体的边界，即**实体边界识别**，然后对于一个确定的”命名实体块“，我们还需要研究它究竟是一个人名、地名、组织名、专有名词还是什么其它的妖孽，即**确定实体类别**。

如何操作呢？英语中的命名实体具有比较明显的形式标志（即实体中的每个词的第一个字母要大写），所以实体边界识别相对容易，任务的重点是确定实体的类别。和英语相比，汉语命名实体识别任务更加复杂，而且相对于实体类别标注子任务，实体边界的识别更加困难。由于汉语不存在分词符，因而命名实体识别的第一步就是我们上一节中所研究的分词。我们可以得出结论：**分词和命名实体识别互相影响**。同时，提取命名实体内部的不同特征也是命名实体识别与分类中可用的手段。这里我们列举一下常用的命名实体识别的方法：

![NERC](C:\Users\tony\Desktop\RDFZ\AI\NLP\NLP#3\NERC.jpg)

### 词性标注

对于研究词性标注，我们还是分类讨论。

先考虑如英语这样的印欧语系的语言。学过一点英语语法就不难发现，英语的单词词中就有着较明显的标注特征。由于笔者英语语法实在是比较烂QAQ，我们来看一个西语中的小例子：西语中绝大多数的动词原形都以 -ar/-er/ir 结尾，而只要在词语中发现一个-mente的后缀，那么这个词语就大概率是一个副词了。我们可以得出结论，在这些语言中，**词语前后缀有着很强的规律性**，用传统的字符串算法，就可以粗略的实现词性的标注。

回到汉语的现实，我们先接着试图寻找这样的规律，然后我们就可以愉快地发现......并没有。。。汉语不光缺乏如形态变化、前后缀特征这样的规律；而且许多常用词还可以表示多种词性；并且汉语还缺乏一种被广泛认可的划分标准。。。真是令人愉快的暴击三连啊hhhhh。只能说，能说中文的人都真不容易啊。

### 依存句法分析

有了前面的铺垫，我们来看一下最重要的句法分析部分。在这里我们来常用的依存句法理论。

依存句法理论的基础是前面的词性标注等步骤，本质上这个理论认为一个句子中存在许多层级的关系，句子中的一些元素影响另一些元素。我们用建房子来打一个比方：

如果建筑师需要建一个摩天楼，那么他一定需要一个很坚实的地基，因为地基支撑起整做大楼。同样，第二层不能脱离第一层而存在，第三层又不能脱离第二层而存在，正所谓没有“空中楼阁”这样的高端操作。在一个句子中同样也是这样的，有一些元素只有在另一种元素出现时才可能会出现，否则这句话就变成了一个空中楼阁。因此依存句法理论认为语句中就存在着元素上的这种依存关系。

如果建筑师想要搭一个柱子，那么他一定需要许多与之相连的结构去限定柱子的运动。柱子有很多种存在的形态，可以竖直、歪着甚至躺着，而建筑师需要用其它的结构去限制它，使得它只能竖直立着。同样，在一句话中，是指某些词的出现只是为了限定其他词的意义。

有了这样的理论，我们便可以一步步剥离句子中的依存关系，进行句法分析，进行语意理解。