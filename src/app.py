import gradio as gr
from rag_core import HuggingFaceAssistant

print("⏳ Initializing RAG Agent...")
bot = HuggingFaceAssistant()
print("✅ Agent ready!")

def predict(message, history):
    for response in bot.answer_stream(message):
        yield response

theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="gray",
)

with gr.Blocks(title="HF RAG Assistant") as demo:
    gr.Markdown(
        """
        # 🦜🤗 Hugging Face Advanced RAG
        **Technical Assistant for:** Transformers, PEFT, Accelerate, TRL, Datasets.
        """
    )
    
    chat_interface = gr.ChatInterface(
        fn=predict,
        examples=[
            "How to use LoRA with 4-bit quantization?",
            "How to create a custom dataset in Datasets library?",
            "Explain Mixed Precision training in Accelerate",
            "What is the SFTTrainer format in TRL?"
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=theme)
