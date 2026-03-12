import ollama
import os

def compare_images_ollama(pre_image_path, post_image_path):

    prompt = """
    You are analyzing two satellite images of the same location.

    The FIRST image is the PRE-disaster image.
    The SECOND image is the POST-disaster image (after a hurricane).

    Compare and contrast the two images and provide:
    1. Visible structural damage
    2. Flooding or water coverage changes
    3. Debris or environmental changes
    4. Overall severity assessment (Low, Moderate, Severe)
    5. A brief summary of key differences

    Be specific and structured in your response.
    """

    try:
        stream = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [pre_image_path, post_image_path]
                }
            ],
            stream=True
        )

        full_response = ""

        for chunk in stream:
            content = chunk["message"].get("content", "")
            print(content, end="", flush=True)
            full_response += content

        print("\n")
        return full_response

    except Exception as e:
        return f"An error occurred: {e}"


images_folder = "/Users/jaimerobles/Desktop/Code/Datasets/images"

image_pairs = {}

for file in os.listdir(images_folder):

    if not file.endswith(".png"):
        continue

    if "_pre_disaster" in file:
        key = file.replace("_pre_disaster.png", "")
        image_pairs.setdefault(key, {})["pre"] = os.path.join(images_folder, file)

    elif "_post_disaster" in file:
        key = file.replace("_post_disaster.png", "")
        image_pairs.setdefault(key, {})["post"] = os.path.join(images_folder, file)


for key, pair in image_pairs.items():

    if "pre" in pair and "post" in pair:
        print(f"\n\n===== Comparing {key} =====\n")

        description = compare_images_ollama(pair["pre"], pair["post"])

        print("\nFinal Collected Response:\n")
        print(description)