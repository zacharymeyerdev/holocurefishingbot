import cv2
import numpy as np
import urllib.request
import pyautogui
import mss

# Define the images (url)
image_urls = {
    'circle': "https://raw.githubusercontent.com/holocurefishingbot/holocurefishingbot/main/images/circle.png",
    'up_arrow': "https://raw.githubusercontent.com/holocurefishingbot/holocurefishingbot/main/images/uparrow.png",
    'down_arrow': "https://raw.githubusercontent.com/holocurefishingbot/holocurefishingbot/main/images/downarrow.png",
    'left_arrow': "https://raw.githubusercontent.com/holocurefishingbot/holocurefishingbot/main/images/leftarrow.png",
    'right_arrow': "https://raw.githubusercontent.com/holocurefishingbot/holocurefishingbot/main/images/rightarrow.png"
}

# Load reference images and calculate Hu Moments
reference_hu_moments = {}
for shape, url in image_urls.items():
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    # Calculate Hu Moments
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    reference_hu_moments[shape] = hu_moments

# Define the region of interest (x, y, width, height)
roi = (1131, 687, 96, 120)

# Define shape descriptors
shapes = {
    'up_arrow': 'w',
    'down_arrow': 's',
    'left_arrow': 'a',
    'right_arrow': 'd',
    'circle': 'space',
}

def compare_shapes(hu_moments1, hu_moments2):
    return np.allclose(hu_moments1, hu_moments2, rtol=1e-5, atol=1e-8)

while True:
    with mss.mss() as sct:
        screen = np.array(sct.shot())
        
    # Extract the region of interest
    roi_image = screen[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Create a binary mask of the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Calculate Hu Moments for the detected contour
        moments = cv2.moments(mask)
        detected_hu_moments = cv2.HuMoments(moments)
        
        # Compare with reference Hu Moments
        for shape, hu_moments in reference_hu_moments.items():
            if compare_shapes(detected_hu_moments, hu_moments):
                key = shapes.get(shape)
                if key:
                    pyautogui.press(key)
                break  # Break if a shape is found to avoid pressing multiple keys
