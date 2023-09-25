import cv2
import numpy as np
import urllib.request
import pyautogui
import mss
import matplotlib.pyplot as plt
import logging
import cProfile
import configparser


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ShapeDetector:
    def __init__(self, reference_hu_moments):
        self.reference_hu_moments = reference_hu_moments

    def detect_shape(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            moments = cv2.moments(mask)
            detected_hu_moments = cv2.HuMoments(moments)

            for shape, hu_moments in self.reference_hu_moments.items():
                if compare_shapes(detected_hu_moments, hu_moments):
                    return shape

        return None


def compare_shapes(hu_moments1, hu_moments2):
    return np.allclose(hu_moments1, hu_moments2, rtol=1e-3, atol=1e-6)  # Adjusted tolerance values


def load_reference_images(image_urls):
    reference_hu_moments = {}
    for shape, url in image_urls.items():
        try:
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

            moments = cv2.moments(image)
            hu_moments = cv2.HuMoments(moments)

            reference_hu_moments[shape] = hu_moments

        except Exception as e:
            logger.error(f"Error loading image for {shape} from {url}: {e}")

    return reference_hu_moments


def main():
    # Load reference images and calculate Hu Moments
    reference_hu_moments = load_reference_images({
        'circle': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/circle.png",
        'up_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/uparrow.png",
        'down_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/downarrow.png",
        'left_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/leftarrow.png",
        'right_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/rightarrow.png"
    })

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

class ShapeDetector:
    def __init__(self, reference_hu_moments):
        self.reference_hu_moments = reference_hu_moments

    def detect_shape(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            moments = cv2.moments(mask)
            detected_hu_moments = cv2.HuMoments(moments)

            for shape, hu_moments in self.reference_hu_moments.items():
                if compare_shapes(detected_hu_moments, hu_moments):
                    return shape

        return None


def main():
    # Load reference images and calculate Hu Moments
    reference_hu_moments = load_reference_images({
        'circle': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/circle.png",
        'up_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/uparrow.png",
        'down_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/downarrow.png",
        'left_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/leftarrow.png",
        'right_arrow': "https://raw.githubusercontent.com/zacharymeyerdev/holocurefishingbot/main/images/rightarrow.png"
    })

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

    # Create a shape detector
    shape_detector = ShapeDetector(reference_hu_moments)

    # Start the fishing bot loop
    while True:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screen = np.array(sct.grab(monitor))

            # Crop the region of interest
            roi_image = screen[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

            # Detect the shape in the region of interest
            shape = shape_detector.detect_shape(roi_image)

            # Press the corresponding key if a shape is detected
            if shape:
                print(f"Shape detected: {shape}")
                key = shapes.get(shape)
                if key:
                    pyautogui.press(key)

        except Exception as e:
            logger.error(e)

            # Show the region of interest image
        cv2.imshow('ROI', roi_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
