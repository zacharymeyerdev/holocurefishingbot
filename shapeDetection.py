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

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def detect_shape(self, image):
        edges = self.preprocess_image(image)
        
        for shape, template in self.templates.items():
            template_edges = self.preprocess_image(template)
            for scale in np.linspace(0.5, 1.5, 10):
                resized_template = cv2.resize(template_edges, None, fx=scale, fy=scale)
                res = cv2.matchTemplate(edges, resized_template, cv2.TM_CCOEFF_NORMED)
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
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            templates[shape] = image

        except Exception as e:
            logger.error(f"Error loading image for {shape} from {url}: {e}")

    return templates

def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def main():
    config = load_config()
    frame_rate = config.getint('DEFAULT', 'frame_rate')
    roi = (
        config.getint('DEFAULT', 'roi_left'),
        config.getint('DEFAULT', 'roi_top'),
        config.getint('DEFAULT', 'roi_width'),
        config.getint('DEFAULT', 'roi_height')
    )
    templates = load_templates({
        'circle': config['Templates']['circle'],
        'up_arrow': config['Templates']['up_arrow'],
        'down_arrow': config['Templates']['down_arrow'],
        'left_arrow': config['Templates']['left_arrow'],
        'right_arrow': config['Templates']['right_arrow']
    })
    print("Templates loaded:", templates)
    
    shapes = {
        'up_arrow': 'w',
        'down_arrow': 's',
        'left_arrow': 'a',
        'right_arrow': 'd',
        'circle': 'space',
    }

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
                logger.debug(f"Detected shape: {shape}")

                # Press the corresponding key if a shape is detected
                if shape:
                    key = shapes.get(shape)
                    if key:
                        pyautogui.press(key)
                        logger.debug(f"Key pressed: {key}")
                
                end_time = time.time()
                detection_time = end_time - start_time
    
                sleep_time = frame_time - detection_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                cv2.imshow('ROI', roi_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
