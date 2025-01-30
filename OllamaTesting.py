import ollama
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import multiprocessing

CONTEXT = "Describe any obstacles you see in front of you and their approximate distance and angle to the camera. Also, describe any objects that could be important reference points. "

GOAL = "Drive the rover to the box"

def camera_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    while not stop_event.is_set():
        ret, frame = cap.read()

        if processing_event.is_set():
            cv2.putText(frame, "Processing...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Camera Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and not processing_event.is_set():  # Spacebar
            print("Processing frame...")
            if not frame_queue.full():
                frame_queue.put(frame)
                trigger_event.set()
                processing_event.set()
        elif key == ord('q'):
            stop_event.set()
    
    cap.release()
    cv2.destroyAllWindows()

def inference_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event):
    while not stop_event.is_set():
        if trigger_event.is_set() and not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imwrite('temp_frame.jpg', frame)
            
            response = ollama.chat(
                model='llava',
                messages=[{
                    'role': 'user',
                    'content': CONTEXT,
                    'images': ['temp_frame.jpg']
                }]
            )
            print(f"Llava Response: {response['message']['content']}")

            response2 = ollama.chat(
                model='deepseek-r1:7b',
                messages=[{
                    'role': 'user',
                    'content': "Given the context that you are an ai in an autonomous rover and you are trying to reach the goal: " + GOAL + "How can the rover achieve this goal if the image in front of it is described as: " + response['message']['content'] + "Give a specific set of rover instructions in the form of 'Forward 10 inches', 'Turn 90 degrees left', etc.",
                }]
            )
            print(f"deepseek Response: {response2['message']['content']}")
            processing_event.clear()
            trigger_event.clear()

def main():
    multiprocessing.set_start_method('spawn')
    frame_queue = Queue(maxsize=1)
    trigger_event = Event()
    processing_event = Event()
    stop_event = Event()
    
    processes = [
        Process(target=camera_process, args=(frame_queue, trigger_event, processing_event, stop_event)),
        Process(target=inference_process, args=(frame_queue, trigger_event, processing_event, stop_event))
    ]
    
    for p in processes:
        p.start()
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        stop_event.set()
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    main()