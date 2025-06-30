import os
from PIL import Image, ImageDraw, ImageFont

def add_water_amrk_to_image(input_dir="input_dir", output_dir="output_dir", water_mark_text="", position=(0, 0), font_size=32):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        # Apply only if file is an image
        if file_name.lower().endswith((".jpg",".jpeg", ".png", ".bmp")):
            image_path = os.path.join(input_dir, file_name)
            original_image = Image.open(image_path)
            width, height = original_image.size
            print(f"Image name: {file_name}")
            print(f"Width: {width}, Height: {height}")
            
            draw_image = ImageDraw.Draw(original_image)
            
            font = ImageFont.load_default(font_size)
            
            _, _, text_width ,text_height  = font.getbbox(water_mark_text)
            
            #Calculate coordinates of text
            x_coordinate = width - text_width - 100 
            y_coordinate = height - text_height - 100 
            
            draw_image.text((x_coordinate, y_coordinate), water_mark_text, "white", font=font)
            
            original_image.save(os.path.join(output_dir, 'water_mark_'+file_name))
            
            
           
add_water_amrk_to_image(water_mark_text="@umind67", font_size=100)
            
            
        

