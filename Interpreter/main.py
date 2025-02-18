import ollama
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import multiprocessing
from RobotCommandInterpreter import RobotCommandInterpreter

CONTEXT = '''You are an AI assisting a robot.  Your primary task is to identify obstacles and a target object.
First, identify the target object.  Then, identify any obstacles that might be in the robot's path.

For each object, provide: name, estimated distance in inches, estimated angle relative to the robot's center based on the location left-right in the image formatted in degrees. If any value is not discernible, respond with 'unknown'

Format the angle so that 0 degrees is directly ahead, negative values are to the left, and positive values are to the right.

Structure your response as follows in json format:
{
"target": {"name": "box", "distance": "12 inches", "angle": "20 degrees"},
"obstacles": [{"name": "obstacle", "distance": "24 inches", "angle": "15 degrees"}, {"name": "obstacle", "distance": "unknown", "angle": "unknown"}]
}
'''

GOAL = "Drive the rover to the trash can"

def show_goal_menu(frame):
    """Display goal menu on frame"""
    menu_frame = frame.copy()
    cv2.putText(menu_frame, "Select Goal:", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(menu_frame, "1: Drive to the box", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(menu_frame, "2: Drive to the trash can", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(menu_frame, "3: Drive to the person", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return menu_frame

def camera_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event): # type: ignore
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    show_menu = False
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        display_frame = frame.copy()

        if processing_event.is_set():
            cv2.putText(display_frame, "Processing...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif show_menu:
            display_frame = show_goal_menu(frame)
        else:
            cv2.putText(display_frame, "Press SPACE to select goal", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Camera Feed', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and not processing_event.is_set():  # Spacebar
            show_menu = True
        elif show_menu and key in [ord('1'), ord('2'), ord('3')]:
            if not frame_queue.full():
                goal = {
                    ord('1'): "Drive the rover to the box",
                    ord('2'): "Drive the rover to the trash can",
                    ord('3'): "Drive the rover to the person"
                }[key]
                print(f"\nSelected goal: {goal}")
                frame_queue.put((frame, goal))
                trigger_event.set()
                processing_event.set()
                show_menu = False
        elif key == ord('q'):
            stop_event.set()
    
    cap.release()
    cv2.destroyAllWindows()

def inference_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event): # type: ignore
    while not stop_event.is_set():
        if trigger_event.is_set() and not frame_queue.empty():
            frame, goal = frame_queue.get()  # Unpack both frame and goal
            cv2.imwrite('temp_frame.jpg', frame)

            response = ollama.chat(
                model='llava',
                messages=[{
                    'role': 'user',
                    'content': "Context: " + CONTEXT + " -- Goal: " + goal,
                    'images': ['temp_frame.jpg']
                }]
            )
            llm_output = response['message']['content']
            print(f"Goal: {goal}")
            print(f"Llava Response: {llm_output}")

            # Initialize the interpreter with the current goal
            interpreter = RobotCommandInterpreter(goal)

            # Generate commands using the JSON interpreter
            commands = interpreter.interpret_json(llm_output)
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