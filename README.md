# ChabBot-PyTorch
A seq2seq based chatbot built with PyTorch, trained with Chinese datasets.
# Dependencies
* 1, PyTorch 0.4.1
* 2, Python 3

# Tests
```
>>> chat("早上好")
'早上好，奚'
>>> chat("下午好")
'请我吃午餐可以吗'
>>> chat("晚上好")
'我的麻麻你啦'
>>> chat("机器人不需要吃午餐")
'那要么的吃太鲜掉毛'
>>> chat("世界上谁最帅")
'何路'
>>> chat("何路是谁")
'傲梦的贝儿！'
>>> chat("贝尔在哪")
'北京西金'
>>> chat("尼玛")
'(￣▽￣)'
>>> chat("上海去过没")
'一海初上啦'
>>> chat("上海")
'看不起外地人的破地方'
>>> chat("上海 南京 北京")
'国帝王死'
>>> chat("南京")
'来旅游，地铁出行是首选'
>>> chat("无锡")
"文龙哥说'可爱的依依在那里"
>>> chat("我是男是女")
'这么可爱一定是男孩纸'
>>> chat("你是男是女")
'我不喜欢这样归类'
>>> chat("我喜欢男还是女")
'喔~较！是女人想simai'
>>> chat("你喜欢男还是女")
'喔~当然是男人，simaimi喜欢搞基'
>>> chat("卧槽")
'不要啊，主人，我很乖的'
>>> chat("卧槽 一只大傻逼")
'就是你种打般'
>>> chat("卧槽 一只智障")
'就知道～'
```

Datasets is 小黄鸡语料。All duplicated question pairs are dropped out as the first duplicated question pairs are kept, the preprocessed dataset has about 260000 question-answer pairs. Note that the used training dataset is not high quality, a good dataset leads to a  much better chatbot, as opensource high quality chinese Q-A dataset is nearly zero in the internet. It is very possible to develop an assistant in a special field using a specialized dataset, if I have the dataset. 
