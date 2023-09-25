import cv2
import time
import numpy as np
import urllib.request
import pyautogui
import mss
import matplotlib.pyplot as plt

# Define the images (url)
image_urls = {
    'circle': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/circle.png",
    'up_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/uparrow.png",
    'down_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/downarrow.png",
    'left_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/leftarrow.png",
    'right_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/rightarrow.png"
}

# Load reference images and calculate Hu Moments
reference_hu_moments = {}
for shape, url in image_urls.items():
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        
        # Debug: Visualize the reference images
        cv2.imshow(shape, image)
        cv2.waitKey(0)
        
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)
        reference_hu_moments[shape] = hu_moments
        
        # Debug: Print Reference Hu Moments
        print(f"{shape} Hu Moments: {hu_moments}")
        
    except Exception as e:
        print(f"Error loading image for {shape} from {url}: {e}")

# Define the region of interest (x, y, width, height)
roi = (1129, 703, 104, 108)

# Define shape descriptors
shapes = {
    'up_arrow': 'w',
    'down_arrow': 's',
    'left_arrow': 'a',
    'right_arrow': 'd',
    'circle': 'space',
}

def compare_shapes(hu_moments1, hu_moments2):
    return np.allclose(hu_moments1, hu_moments2, rtol=1e-3, atol=1e-6)  # Adjusted tolerance values

while True:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screen = np.array(sct.grab(monitor))
    
    plt.imshow(screen)
    plt.title('Screen')
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()
    cv2.waitKey(1)

    roi_image = screen[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.imshow('ROI', roi_image)
    cv2.waitKey(1)

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # Adjusted threshold value
    cv2.imshow('Threshold', thresh)
    cv2.waitKey(1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    for contour in contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imshow('Mask', mask)
        cv2.waitKey(1)

        moments = cv2.moments(mask)
        detected_hu_moments = cv2.HuMoments(moments)
        print(f"Detected Hu Moments: {detected_hu_moments}")

        for shape, hu_moments in reference_hu_moments.items():
            diff = np.abs(detected_hu_moments - hu_moments)  # Debug: Compare Hu Moments Manually
            print(f"Difference with {shape}: {diff}")
            
            if compare_shapes(detected_hu_moments, hu_moments):
                print(f"Detected shape: {shape}")
                key = shapes.get(shape)
                if key:
                    print(f"Pressing key: {key}")
                    pyautogui.press(key)
                break