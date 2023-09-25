import cv2
import numpy as np
import pyvirtualcam
from PIL import Image, ImageFont, ImageDraw
import multiprocessing, time
import zmq

########################################################################
# SETTINGS #
WIDTH = 1920 // 2
HEIGHT = 1080 // 2
FPS = 60

RESIZED_WIDTH = 240     # 256, 240, 160, 112, 80
RESIZED_HEIGHT = 135     # 144, 135, 90, 63, 45
FONT_SIZE = 6      # 14, 14, 20, 28, 40
# FONT_ALPHA = 1         # 1, 1, 2, 3, 3
ROW_SPACING = HEIGHT / RESIZED_HEIGHT

LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 120])

CAP_INPUT = 0 # 0 for camera
########################################################################
        
def sender():
    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, print_fps=True) as cam:
        print(f'Using virtual camera: {cam.device}')
        context = zmq.Context()
        puller = context.socket(zmq.PULL)
        puller.bind("tcp://*:5558")
        while True:
            try:
                try:
                     result_o = puller.recv(flags=zmq.NOBLOCK)
                except zmq.Again as e:
                     continue
                except KeyboardInterrupt:
                     break
                # result_o = puller.recv()
                
                deserialized_image = np.frombuffer(result_o, dtype=np.uint8)
                deserialized_image = deserialized_image.reshape(HEIGHT, WIDTH)
                
                r_channel = np.zeros_like(deserialized_image)
                g_channel = deserialized_image.copy()
                b_channel = np.zeros_like(deserialized_image)

                # Merge the three channels to form an RGB image
                rgb_image = cv2.merge((b_channel, g_channel, r_channel))
                
                cam.send(rgb_image)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    context = zmq.Context()
    pusher = context.socket(zmq.PUSH)
    pusher.bind("tcp://*:5559")  # Publisher binds to a specific address and port
    # pusher.setsockopt(zmq.SNDHWM, 21)
    
    cap = cv2.VideoCapture(CAP_INPUT)
    cap_width  = cap.get(3)
    cap_height = cap.get(4)
    
    process = multiprocessing.Process(target=sender, args=())
    process.start()
    
    # TODO: Gives time for process to load up. This prevents the hyper FPS when starting from video. This can
    # be resolved by having the process running as a standalone service.
    time.sleep(2)
    
    while True:
        try:
            ret, frame = cap.read()      
            image = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mask = cv2.inRange(image, LOWER_BLUE, UPPER_BLUE)
            image[mask != 0] = [0, 0, 0]
            reduced = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            serialized_bytes = reduced.tobytes()
            pusher.send(serialized_bytes)
        except Exception as e:
            print(e)
            break
    cap.release()
    cv2.destroyAllWindows()