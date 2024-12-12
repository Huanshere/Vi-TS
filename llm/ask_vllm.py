import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# detail 参数会将图片分辨率向上取整到 28 倍数， 224 * 448 消耗 128tokens，对应 0.05 分 rmb
def encode_image(image_path: str) -> dict:
    """Encode a single image to base64 and return message dict."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }

def ask_vllm(image_paths: list, prompt: str, model:str="pro") -> str:
    # Prepare messages with multiple images
    model = "Qwen/Qwen2-VL-72B-Instruct" if model == "pro" else "Pro/Qwen/Qwen2-VL-7B-Instruct"
    messages = [encode_image(path) for path in image_paths]
    
    # Add the text prompt
    messages.append({
        "type": "text",
        "text": prompt
    })
    
    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages}],
        stream=True
    )
    
    # Collect and return the response
    full_response = ""
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            full_response += chunk_message
            print(chunk_message, end='', flush=True)
    
    return full_response

if __name__ == "__main__":
    image_paths = ["llm/demo.png"]
    prompt = "详细描述图中人物的所有服装，并估算服装热阻，单位 clo"
    ask_vllm(image_paths, prompt, model="pro")