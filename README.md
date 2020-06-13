# 基于BERT的无监督分词和句法分析

文章 [Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT](https://arxiv.org/abs/2004.14786) 所提的方法在中文上的简单验证。

- 博客介绍：https://kexue.fm/archives/7476
- 原作实现：https://github.com/LividWo/Perturbed-Masking

# 演示

无监督分词效果：
```python
[u'习近平', u'总书记', u'6月', u'8日', u'赴', u'宁夏', u'考察', u'调研', u'。', u'当天', u'下午', u'，他先后', u'来到', u'吴忠', u'市', u'红寺堡镇', u'弘德', u'村', u'、黄河', u'吴忠', u'市城区段、', u'金星', u'镇金花园', u'社区', u'，', u'了解', u'当地', u'推进', u'脱贫', u'攻坚', u'、', u'加强', u'黄河流域', u'生态', u'保护', u'、', u'促进', u'民族团结', u'等', u'情况', u'。']

[u'大肠杆菌', u'是', u'人和', u'许多', u'动物', u'肠道', u'中最', u'主要', u'且数量', u'最多', u'的', u'一种', u'细菌']

[u'苏剑林', u'是', u'科学', u'空间', u'的博主']

[u'九寨沟', u'国家级', u'自然', u'保护', u'区', u'位于', u'四川', u'省', u'阿坝藏族羌族', u'自治', u'州', u'南坪县境内', u'，', u'距离', u'成都市400多公里', u'，', u'是', u'一条', u'纵深', u'40余公里', u'的山沟谷', u'地']
```

无监督句法分析：

<--img src="https://kexue.fm/usr/uploads/2020/06/1080117526.png" width=560>

# 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
