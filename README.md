# BatchLLM_Usage
Generate Batch Clinical Notes utilizing LLM

### Requirement

```
torch >= 1.3, <2.0
omegaconf
InstructorEmbedding
openai
langchain
tiktoken
nltk
rice
sentence_transformers
datasets
sentencepiece != 0.1.92
```



现在需要做的工作：

主要是仿照Ex1， Ex2的方式，去写Ex3.

### EX2的主要内容

使用batch的方式，利用chatgpt-api接口，尝试给1个example的同时给batch_size个对话样本，看chatgpt生成的报告效果如何。

Prompt示例：

```python
prompt = PromptTemplate(
        input_variables=["examples", "dialogue1", "dialogue2"],
        template=
        """Write clinical note reflecting doctor-patient dialogue. 
        I provide you with two dialogues(1,2) and you should generate two clinical notes. 
        DIALOGUES1: [{dialogue1}]
        DIALOGUES2: [{dialogue2}]
        EXAMPLES: [{examples}]
        Write clinical note reflecting doctor-patient dialogue. Use the example notes above to decide the structure of the clinical note. Do not make up information.
        I want you to give your output in a markdown table where the first column is the id of dialogues and the second is the clinical note for each dialogue.
        """
    )
```

### Ex3的主要内容：

可能因为生成文本过长的原因效果并不好，所以现在先让chatgpt分章节生成，即针对某个章节单独生成医疗报告的内容。

在本次Ex3中先生成章节'HISTORY OF PRESENT ILLNESS'的内容吧。

使用batch的方式，利用chatgpt-api接口，尝试给1个 HISTORY OF PRESENT ILLNESS example的同时给batch_size个对话样本，看chatgpt生成的报告效果如何。

batch_size可以取4先试一试~

