import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# 创建多个 Hugging Face 管道
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")

def multi_task_nlp(text, task):
    """
    根据选择的任务执行不同的 NLP 分析
    """
    if not text:
        return "请输入文本"
    
    try:
        if task == "情感分析":
            results = sentiment_analyzer(text, max_length=512, truncation=True)
            result = results[0]
            emotion = "😊 积极" if result['label'] == "POSITIVE" else "😔 消极"
            return f"情感: {emotion}\n置信度: {result['score']:.2%}"
            
        elif task == "文本摘要":
            # 限制输入长度以避免超出模型限制
            if len(text.split()) > 1024:
                return "文本太长，请输入少于 1024 个单词的文本"
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
            
        elif task == "英译中":
            translation = translator(text, max_length=512)
            return translation[0]['translation_text']
            
    except Exception as e:
        return f"处理时出错: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Hugging Face + Gradio 多功能 NLP 演示
    
    这个应用展示了如何使用不同的 Hugging Face 模型来完成各种 NLP 任务。
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="在这里输入文本...",
                lines=5
            )
            task = gr.Radio(
                ["情感分析", "文本摘要", "英译中"],
                label="选择任务",
                value="情感分析"
            )
            submit_btn = gr.Button("分析", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(
                label="分析结果",
                lines=5
            )
    
    # 示例
    gr.Examples(
        examples=[
            ["I absolutely love this new smartphone! The camera quality is incredible.", "情感分析"],
            ["The quick brown fox jumps over the lazy dog. This is a pangram sentence containing all letters of the alphabet.", "英译中"],
            ["Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now perform tasks that previously required human intelligence, from recognizing speech and images to making complex decisions.", "文本摘要"]
        ],
        inputs=[input_text, task],
        outputs=output,
        fn=multi_task_nlp
    )
    
    # 绑定事件
    submit_btn.click(
        fn=multi_task_nlp,
        inputs=[input_text, task],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 使用的模型:
    - **情感分析**: distilbert-base-uncased-finetuned-sst-2-english
    - **文本摘要**: facebook/bart-large-cnn
    - **英译中**: Helsinki-NLP/opus-mt-en-zh
    """)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )