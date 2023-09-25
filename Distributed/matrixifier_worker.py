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

# TODO: Fix font sizes due to reduced resolution
RESIZED_WIDTH = 240     # 256, 240, 160, 112, 80
RESIZED_HEIGHT = 135     # 144, 135, 90, 63, 45
FONT_SIZE = 6      # 14, 14, 20, 28, 40
# FONT_ALPHA = 1         # 1, 1, 2, 3, 3
ROW_SPACING = HEIGHT / RESIZED_HEIGHT

LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 120])

CAP_INPUT = "C:/Users/johne/Desktop/sample2.mkv" # 0 for camera
########################################################################

def convert_row_to_ascii(row):
    ORDER = (' ', '.', "'", ',', ':', ';', 'c', 'l',
             'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')
    return tuple(ORDER[int(x / (255 / 16))] for x in row)[::-1]

def convert_to_ascii(input_grays):
    return tuple(convert_row_to_ascii(row) for row in input_grays)

def worker():
    monospace = ImageFont.truetype("./Fonts/ANDALEMO.ttf", FONT_SIZE)
    # _, top, _, bottom = monospace.getbbox(" .',:;clxokXdO0KN")
    # SPACING = ROW_SPACING - (bottom - top) - FONT_ALPHA
    frame_background = np.zeros((HEIGHT, WIDTH), np.uint8)
    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.connect("tcp://192.168.1.10:5559")
    context = zmq.Context()
    pusher = context.socket(zmq.PUSH)
    pusher.connect("tcp://192.168.1.10:5558")
    # TODO: Does this do anything?
    pusher.setsockopt(zmq.SNDHWM, 1) 
    while True:
        try:
            start_time = time.time_ns()
            try:
                reduced = puller.recv(flags=zmq.NOBLOCK)
            except zmq.Again as e:
                time.sleep(0.01)
                continue
            except KeyboardInterrupt:
                break
            # reduced = puller.recv()
            deserialized_image = np.frombuffer(reduced, dtype=np.uint8)
            deserialized_image = deserialized_image.reshape(RESIZED_HEIGHT, RESIZED_WIDTH)

            im_p = Image.fromarray(frame_background)
            draw = ImageDraw.Draw(im_p)
            
            # Plug in reduced resolution numpy array for ascii converter func

            converted = convert_to_ascii(deserialized_image)
            # converted = convert_to_ascii(reduced)
            
            # multiline_string = ''
            # for row in converted:
            #     multiline_string += ''.join(row) + '\n'
            # draw.multiline_text((0,0), multiline_string, (0,255,0), spacing=SPACING, font=monospace)
            
            
            for index, row in enumerate(converted):
                draw.text((0, index * (ROW_SPACING)),''.join(row),(255),font=monospace)
            
            # Convert back to OpenCV image
            result_o = np.array(im_p)
            received_data = np.frombuffer(result_o, dtype=np.int32)
            serialized_bytes = received_data.tobytes()
            try:
                pusher.send(serialized_bytes, zmq.NOBLOCK)
            except Exception as e:
                print(f'Internal: {e}')
            end_time = time.time_ns()
            print(end_time - start_time)

        except Exception as e:
            print(f'Sender Error: {e}')

if __name__ == "__main__":
    
    available_cores = multiprocessing.cpu_count()
    # process_cores = 21 if available_cores >= 24 else available_cores - 3
    process_cores = 6
    assert process_cores >= 4
    for i in range(process_cores):
        p = multiprocessing.Process(target=worker)
        p.start()
    
    while True:
        try:
            time.sleep(1)
        except Exception as e:
            print(e)