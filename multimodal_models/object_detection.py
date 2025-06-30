import ollama
import re
import streamlit as st
import io
from PIL import Image, ImageDraw

class ObjectDetector():
    MIN_IMG_SIZE = 256
    def __init__(self):
        self.model = 'qwen2.5vl'
        self.img_file = None #St uploaded file type
        
    def scan_image(self, image_bytes):
        try:
            message = [{'role': 'user', 'content': """Give me a detailed description 
                                of the following image and provide me a bounding box of all 
                                object that you are able to detect, provide the bounding box in JSON format, Please use this format
                                 ## Detailed description
                                 ## Bounding Boxes JSON format
                                 (only generate a bounding box for main object)
                                 ## Description of bounding boxes
                                 (Use the label you assign in the Bounding Boxes section to describe the bounding box)
                                 """, "images" : [image_bytes]}]
            results = ollama.chat(model=self.model, messages=message)
            if results:
                print(results.message['content'])
                st.success("Image properly scanned")
                return results.message['content']
        except Exception as e:
            st.error(f"Error scanning image: {e}")
        
    def get_detailed_description(self, response):
        pattern = r"## Detailed description\s+[\w\.,\s-]+"
        match = re.search(pattern=pattern, string=response)
        if match:
            detailed_description = response[match.start(): match.end()]
            detailed_description = re.sub(pattern=r"## Detailed description\s+", repl="", string=detailed_description)
            st.success("Detailed description obtained")
            return detailed_description
        st.warning("No description provided by LLM")
        return None
    
    def get_bbox_list(self, response):
        pattern =r"```json\s+(\[[\s\{\w\":,\[\]\}]+\s+\])\s+```"
        match = re.search(pattern=pattern, string=response)
        if match:
            st.success("Bounding boxes obtained")
            return eval(match.group(1))
        st.warning("No bounding box provided by LLM")
        return None
    
    def draw_bboxes(self, img, bbox_list, max_bboxes = 6):
        colors = ["red", "blue", "white", "green", "yellow", "purple"]
        draw = ImageDraw.Draw(img)
        bbox_list = bbox_list[:min(max_bboxes, len(bbox_list))]
        outcolor = {}
        for i, bbox in enumerate(bbox_list):
            coordinates = bbox["bbox_2d"]
            label = bbox["label"]
            draw.rectangle(xy=coordinates, outline=colors[i], width=3)
            outcolor[label] = colors[i]
        return img, outcolor
    
    def resize_attached_img(self, img):
        new_width = max(img.width // 6, self.MIN_IMG_SIZE)
        new_height = max(img.height //6, self.MIN_IMG_SIZE)
        resized_img = img.resize((new_width, new_height))
        return resized_img
    
    def process_image(self):
        # Get image bytes
        try:
            image_bytes = self.img_file.read()
            image = Image.open(io.BytesIO(image_bytes)) # PIL image
            resized_img = self.resize_attached_img(image)
            response = self.scan_image(image_bytes)
            if response:
                detailed_description = self.get_detailed_description(response)
                bboxes = self.get_bbox_list(response)
                if bboxes:
                    draw_img, out_color = self.draw_bboxes(resized_img, bboxes, 5)
                    return draw_img, out_color, detailed_description
                else:
                    return resized_img, None, detailed_description
            else:
                st.error("Not response got from LLM")
            
        except Exception as e:
            st.error(f"Error occur while processing image: {e}")
        
test_string = """
## Detailed description
The image depicts a white cat with striking blue eyes sitting in front of a window. The cat has a fluffy coat and is positioned in a relaxed posture, with its front paws slightly apart. The background features a window with a curtain, through which some greenery is visible, suggesting a natural setting outside. There is also a potted plant to the right of the cat, adding to the indoor setting. The lighting in the image is soft and warm, creating a serene and cozy atmosphere.

## Bounding Box JSON format
```json
[
    {
        "bbox_2d": [142, 2, 476, 587],
        "label": "White cat with blue eyes"
    },
    {
        "bbox_2d": [0, 0, 476, 587],
        "label": "Indoor setting with window and curtain"
    },
    {
        "bbox_2d": [375, 200, 476, 350],
        "label": "Potted plant"
    }
]
```

## Description of bounding box
- The first bounding box (`[142, 2, 476, 587]`) encloses the white cat with blue eyes, highlighting its fluffy coat and relaxed posture.
- The second bounding box (`[0, 0, 476, 587]`) encompasses the entire indoor setting, including the window, curtain, and the potted plant.
- The third bounding box (`[375, 200, 476, 350]`) specifically highlights the potted plant to the right of the cat."""

if __name__ == "__main__":
    st.title("🤖 Image scanner and descriptor")

    # Initialize session state
    if "object_detector" not in st.session_state:
        st.session_state.object_detector = ObjectDetector()
    if "detailed_description" not in st.session_state:
        st.session_state.detailed_description = None
    if "draw_img" not in st.session_state:
        st.session_state.draw_img = None
    if "last_processed_img" not in st.session_state:
        st.session_state.last_processed_img = None
        
    image_file = st.file_uploader(label="Insert an image do you want me to scan and describe ...", 
                                type=["png", "jpeg", "jpg"], 
                                accept_multiple_files=False)
    
    if image_file and image_file.name != st.session_state.last_processed_img:
        with st.spinner("Scanning image ..."):
            st.session_state.object_detector.img_file = image_file
            st.session_state.last_processed_img = image_file.name
            draw_img , _, detailed_desc = st.session_state.object_detector.process_image()
            st.session_state.draw_img = draw_img
            st.session_state.detailed_description = detailed_desc
            
    if st.session_state.draw_img and st.session_state.detailed_description:
        # Display description
        st.markdown("### 📝 Detailed description:")
        st.write(st.session_state.detailed_description)  
        
        # Display draw image
        st.image(st.session_state.draw_img, caption="Image with detected objects", use_container_width=True)      