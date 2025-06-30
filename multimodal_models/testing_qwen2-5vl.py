import ollama
from PIL import Image, ImageDraw

#img = Image.open('cat.png')
#scaled_factor = 2
#resized_img = img.resize((img.width // scaled_factor, img.height // scaled_factor))
##
#resized_img.save('cat_resized.png')
#
#with open('cat_resized.png', mode='rb') as image:
#    image_bytes = image.read()
#
#messages = [{'role': 'user', 'content': """Give me a detailed description 
#                    of the following image and provide me a bounding box of all 
#                    object that you are able to detect, provide the bounding box in JSON format, Please use this format
#                     ## Detailed description
#                     ## Bounding Box JSON format
#                     ## Description of bounding box 
#                     """, "images" : [image_bytes]}]
#print("Generating response ...")
#response = ollama.chat(model="qwen2.5vl", messages=messages)
img = Image.open('cat_resized.png')
draw = ImageDraw.Draw(img)
rectangle_coords = [142, 2, 476, 587]
draw.rectangle(rectangle_coords, outline='red', width=3)

img.save('bbox_cat.png')
#print(response.message['content'])