from PIL import Image
import numpy as np
from typing import Union

def resize_image(image:Union[str,np.ndarray], new_width:int, new_length:int, pad_clr:tuple) -> Image.Image:
    """
    Resize an image to fit within a specified width and length, maintaining the aspect ratio.
    If the image does not fit within the specified dimensions, it will be padded with a specified color.
    This is not the same as just padding an image, because the old image will always try to maximize its area 
    in the new image.
    
    image: Image to be resized
    new_width: New width of the image
    new_length: New length of the image
    pad_clr: Color to pad the image with if it does not fit within the specified dimensions
    
    Returns: Resized image as a PIL image.
    """
    #Open Image as either string or numpy array
    if isinstance(image, str):
        img = Image.open(image) 
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError("Unsupported image type")
    
    orig_width, orig_height = img.size

    # Compute scaling factor to fit within target box
    scale = min(new_width / orig_width, new_length / orig_height)
    resized_width = int(orig_width * scale)
    resized_height = int(orig_height * scale)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    # Determine mode and pad color
    mode = img.mode
    if mode == 'L':
        pad_color = pad_clr[0] if isinstance(pad_clr, (tuple, list)) else pad_clr
    else:
        pad_color = pad_clr
        
    # Create new image and paste resized image onto center
    new_img = Image.new(mode, (new_width, new_length), pad_color)
    paste_x = (new_width - resized_width) // 2
    paste_y = (new_length - resized_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    """
    #Old algorithm:
    width, length = img.size
    img_aspect_ratio = width/length #Calculate aspect ratio
    new_img = Image.new('RGB', (new_width, new_length), pad_clr) #Create background image
    
    shortest = min(new_width, new_length) #Find smallest dimension

    if shortest == new_width: #Width is shortest
        length2 = new_width/img_aspect_ratio
        img = img.resize((new_width, int(length2))) #Resize to smallest dim while keeping aspect ratio
        new_img.paste(img, (0, int((new_length-length2)//2))) #Paste onto background
    elif shortest == new_length: #Length is shortest
        width2 = new_length * img_aspect_ratio
        img = img.resize((int(width2), new_length)) #Resize to smallest dim while keeping aspect ratio
        new_img.paste(img, (int((new_width-width2)//2), 0)) #Paste onto background
    else: #Width and length are the same
        dim1 = abs(max(new_width, new_length)-max(width, height))/max(new_width, new_length)
        dim2 = min(new_width, new_length) 
    """
    return new_img

# Example usage
# import cv2
# import numpy as np
# image = resize_image('drone.jpg', 300, 700, (120, 120, 120))
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()