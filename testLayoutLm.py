from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
from PIL import Image

# 1. Tell the computer which LayoutLM puppy we want to use
model_name = "microsoft/layoutlmv3-large-uncased" # This is a popular and capable LayoutLM model
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_name)

# 2. Show the puppy the picture book (your document image)
image_path = "IMG_5032.jpg"
image = Image.open(image_path).convert("RGB")

# Question
question = "How much is Activation/Upgrade Fee?"

# 4. Prepare the question and the image for the puppy
inputs = processor(image, question, return_tensors="pt")

# 5. Ask the puppy and get its answer
outputs = model(**inputs)

# 6. The puppy gives us an answer! Let's see what it says
answer = processor.decode(outputs.logits.argmax(-1), skip_special_tokens=True)
print(f"Question: {question}")
print(f"Answer: {answer}")

# --- For Localization (Finding Where the Answer Is) ---
print("\n--- Localization Information (More Advanced) ---")
if hasattr(outputs, 'start_logits') and hasattr(outputs, 'end_logits'):
    start_index = outputs.start_logits.argmax(-1).item()
    end_index = outputs.end_logits.argmax(-1).item()

    input_ids = inputs.input_ids.squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    answer_tokens = tokens[start_index : end_index + 1]
    print(f"Answer Tokens: {answer_tokens}")
else:
    print("This model might not directly provide detailed localization information in this way.")