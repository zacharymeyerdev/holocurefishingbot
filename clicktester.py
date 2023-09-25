from pynput import mouse
import pyautogui

def on_click(x, y, button, pressed):
    if pressed:
        print(f'You clicked at point: ({x}, {y})')

# Start the mouse listener
with mouse.Listener(on_click=on_click) as listener:
    listener.join()