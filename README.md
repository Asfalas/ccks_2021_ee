# ccks 2021 面向通信领域的过程类知识抽取（一）事件抽取
用于技术方案分享和Todo整理

## Ideas
- 当前模型：（简单联合抽取模型）
    - argument mention detection
        - 使用不带标签的BIO序列标注抽取
    - trigger mention detection
        - 使用带event_type标签的BIO序列标注抽取
    - argument role classification
        - argument mention embedding 拼接 trigger mention embedding 拼接 event type embedding 直接分类
        - 训练时使用golden mention & event type
## Todo
