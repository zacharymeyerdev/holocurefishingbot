import cv2
import numpy as np
import urllib.request
import pyautogui
import mss
import logging
import configparser
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ShapeDetector:
    def __init__(self, templates):
        self.templates = templates

    def detect_shape(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=(3,3))  # Opening Morphology Operation
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for shape, template in self.templates.items():
            res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.8:  # Threshold, adjust accordingly
                return shape

        return None

def load_templates(image_urls):
    templates = {}
    for shape, url in image_urls.items():
        try:
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            templates[shape] = image

        except Exception as e:
            logger.error(f"Error loading image for {shape} from {url}: {e}")

    return templates

def main():
    # Load reference images and calculate Hu Moments
    templates = load_templates({
        'circle': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/circle.png",
        'up_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/uparrow.png",
        'down_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/downarrow.png",
        'left_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/leftarrow.png",
        'right_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/rightarrow.png"
    })
    print("Templates loaded:", templates)

    shape_detector = ShapeDetector(templates)
    
    # Define the region of interest (x, y, width, height)
    roi = (1029, 703, 204, 108)
    shapes = {
        'up_arrow': 'w',
        'down_arrow': 's',
        'left_arrow': 'a',
        'right_arrow': 'd',
        'circle': 'space',
    }

    frame_rate = 30  # e.g., 30 frames per second
    frame_time = 1.0 / frame_rate  # time for one frame in seconds

    shape_detector = ShapeDetector(templates)

    try:
        with mss.mss() as sct:
            while True:
                start_time = time.time()

                # Capturing and processing the screen
                monitor = {"top": roi[1], "left": roi[0], "width": roi[2], "height": roi[3]}
                screen = np.array(sct.grab(monitor))
                roi_image = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)  # If the image is BGRA, convert it to BGR
                
                # Detect the shape in the region of interest
                shape = shape_detector.detect_shape(roi_image)
                print("Detected shape:", shape)

                # Press the corresponding key if a shape is detected
                if shape:
                    key = shapes.get(shape)
                    print("Shape detected: {shape}")
                    if key:
                        pyautogui.press(key)
                        print("Key pressed:", key)

                # Show the region of interest image
                cv2.imshow('ROI', roi_image)
                cv2.waitKey(1)
                print("ROI Image:", roi_image)  # Add this line

                # Calculate the dynamic delay
                elapsed_time = time.time() - start_time
                sleep_time = max(frame_time - elapsed_time, 0)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()