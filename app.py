import gradio as gr
from transformers import pipeline

# 初始化 Hugging Face 情感分析模型
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """
    使用 Hugging Face 模型分析文本情感
    """
    if not text:
        return "请输入文本进行分析"
    
    results = classifier(text)
    result = results[0]
    
    # 格式化输出
    label = result['label']
    score = result['score']
    
    emotion = "😊 积极" if label == "POSITIVE" else "😔 消极"
    confidence = f"{score:.2%}"
    
    return f"情感分析结果: {emotion}\n置信度: {confidence}"

# 创建 Gradio 界面
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="输入文本",
        placeholder="在这里输入要分析的文本...",
        lines=3
    ),
    outputs=gr.Textbox(label="分析结果"),
    title="情感分析演示",
    description="使用 Hugging Face 的 DistilBERT 模型进行文本情感分析",
    examples=[
        ["I love this product! It's amazing."],
        ["This is terrible, I'm very disappointed."],
        ["The weather is nice today."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    # 启动 Gradio 应用
    iface.launch(
        share=False,  # 设置为 True 可以生成公共链接
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860  # 默认端口
    )