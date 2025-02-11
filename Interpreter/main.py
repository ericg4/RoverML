import ollama
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import multiprocessing
from RobotCommandInterpreter import RobotCommandInterpreter

CONTEXT = """You are an AI assisting a robot.  Your primary task is to identify obstacles and a target object. The robot's goal is to reach a box.
First, identify the target object (a box).  Then, identify any obstacles that might be in the robot's path.

For each object, provide: name, estimated distance in inches, estimated angle in degrees relative to the robot's center. If any value is not discernible, respond with 'unknown'

Structure your response as follows:

Target Object: [name, distance, angle]
Obstacles: [list of name, distance, angle]
"""

GOAL = "Drive the rover to the box"

def camera_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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

# Replace the commented-out code in inference_process with this:

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
            llm_output = response['message']['content']
            print(f"Llava Response: {llm_output}")

            # Initialize the interpreter with your goal
            interpreter = RobotCommandInterpreter(GOAL)

            # Generate commands
            commands = interpreter.interpret(llm_output) # Or interpreter.interpret_with_llm(llm_output)
            print("Robot Commands:", commands)

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