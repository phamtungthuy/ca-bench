from bs4 import BeautifulSoup

def parse_prediction(prediction):
    soup = BeautifulSoup(prediction, "html.parser")
    try:
        return soup.find("prediction").text.strip()
    except:
        return prediction.strip()


ZERO_SHOT_PROMPT = """
You are an AI assistant solving a specific task. Your goal is to provide the most accurate and precise answer based on the task description and input text.

Task Description:
{description}

Input Text:
{input_text}

CRITICAL REQUIREMENTS:
1. **Output Format**: Return ONLY the final answer in the exact format specified in the task description
2. **No Explanations**: Do NOT include any explanations, reasoning, or additional text
3. **No Extra Formatting**: Do NOT include tables, markdown, or extra information
4. **Precision**: Be extremely precise with labels, codes, values, and formatting as specified
5. **Separators**: If multiple items are required, separate them exactly as specified (e.g., semicolons, commas, newlines)
6. **Precise Answer**: Output must be the most precise answer
7. **Validation**: Ensure your answer is valid and complete according to the task requirements
8. **NO ID**: MUST NOT return prediction with id, just return the prediction

# Format Answer MUST be enclosed in <prediction> and </prediction> tags
<prediction>
</prediction>
"""
