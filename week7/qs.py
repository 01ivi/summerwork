# 1. 导入必要库（合并重复导入，保持简洁）
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)
import torch

# 2. 加载并处理数据集
squad = load_dataset("squad", split="train[:5000]")  # 取前5000条样本
squad = squad.train_test_split(test_size=0.2)  # 划分训练集（80%）和验证集（20%）
# print(squad["train"][0])  # 可选：取消注释查看样本结构，不影响运行

# 3. 加载分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# 4. 定义数据预处理函数
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,  # 最大序列长度
        truncation="only_second",  # 只截断上下文（不截断问题）
        return_offsets_mapping=True,  # 返回字符偏移量（用于定位答案位置）
        padding="max_length",  # 按最大长度填充
    )

    offset_mapping = inputs.pop("offset_mapping")  # 弹出偏移量，后续处理
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]  # 答案起始字符位置
        end_char = start_char + len(answer["text"][0])  # 答案结束字符位置
        sequence_ids = inputs.sequence_ids(i)  # 区分问题（0）和上下文（1）的标识

        # 定位上下文在输入序列中的起始和结束位置
        idx = 0
        while sequence_ids[idx] != 1:  # 找到上下文起始点（sequence_ids=1）
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:  # 找到上下文结束点（sequence_ids≠1）
            idx += 1
        context_end = idx - 1

        # 若答案不在上下文中，标记为(0,0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 找到答案对应的起始token位置
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # 找到答案对应的结束token位置
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions  # 答案起始token位置
    inputs["end_positions"] = end_positions  # 答案结束token位置
    return inputs

# 5. 应用预处理（移除原始字段，保留模型需要的特征）
tokenized_squad = squad.map(
    preprocess_function,
    batched=True,
    remove_columns=squad["train"].column_names  # 移除原始文本字段（question、context等）
)

# 6. 定义数据整理器（用于批量处理数据）
data_collator = DefaultDataCollator()

# 7. 加载预训练模型（问答任务专用）
model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

# 8. 配置训练参数（核心修改：减小批次大小，确保不超内存）
training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",  # 模型保存路径
    eval_strategy="epoch",  # 每个epoch结束后验证
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=8,  # 训练批次大小（从16减小到8，避免内存不足）
    per_device_eval_batch_size=8,  # 验证批次大小
    num_train_epochs=3,  # 训练轮数
    weight_decay=0.01,  # 权重衰减（防止过拟合）
    logging_dir="./logs",  # 日志路径（可选，便于调试）
    logging_steps=50,  # 每50步打印一次日志
)

# 9. 初始化Trainer（移除无效参数processing_class）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],  # 训练集
    eval_dataset=tokenized_squad["test"],  # 验证集
    data_collator=data_collator,  # 数据整理器
)

# 10. 开始训练
trainer.train()

# 11. 训练后显式保存分词器（关键：确保推理时能加载）
tokenizer.save_pretrained("my_awesome_qa_model")

# 12. 推理（使用训练好的模型）
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."

# 加载训练好的分词器和模型
tokenizer_infer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
model_infer = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")

# 处理输入
inputs = tokenizer_infer(question, context, return_tensors="pt")

# 模型预测
with torch.no_grad():  # 关闭梯度计算，节省内存
    outputs = model_infer(**inputs)

# 解析答案（取logits最大的位置）
answer_start_idx = outputs.start_logits.argmax()
answer_end_idx = outputs.end_logits.argmax()

# 解码答案token
predicted_answer = tokenizer_infer.decode(
    inputs.input_ids[0, answer_start_idx:answer_end_idx + 1]
)

print("预测答案：", predicted_answer)  # 预期输出："13"