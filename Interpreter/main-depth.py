import ollama
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import multiprocessing
from RobotCommandInterpreter import RobotCommandInterpreter
import torch
import depth_pro
from PIL import Image
import matplotlib.pyplot as plt

def get_torch_device():
    print("Detecting available computing devices...")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"CUDA device found: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon MPS device found")
    else:
        print("Using CPU for processing (this may be slower)")
    return device

print("Starting RoverML with Depth Perception...")
print("Loading depth estimation model...")

# Initialize depth model globally
try:
    depth_model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    depth_model.eval()
    print("✓ Depth model loaded successfully")
except Exception as e:
    print(f"! Error loading depth model: {str(e)}")
    print("  Falling back to camera-only mode")
    depth_model = None

print("Initializing system components...")

# Modify the CONTEXT to include positional information
CONTEXT = '''You are an AI assisting a robot. Your primary task is to identify obstacles and a target object using both a regular image and its depth map.
The depth map shows distance information: BLUE areas are FARTHEST from the robot, RED areas are CLOSEST to the robot.

I've calculated exact distance values for each region of the image, which I'll provide to you.
For each region, I will give:
- Position (e.g., "top-left", "middle-center", "bottom-right", "bottom-center", "bottom-left", "top-center", "top-right", "middle-left", "middle-right")
- Exact distance in inches
- Angle relative to center

First, identify the target object. Then, identify any obstacles that might be in the robot's path.

IMPORTANT: For each object you identify, you MUST:
1. name: state what the object is
2. region: specify which region contains this object (e.g., "top-left", "middle-center", "bottom-right", "bottom-center", "bottom-left", "top-center", "top-right", "middle-left", "middle-right")
3. angle: state the angle relative to center (e.g., "-15 degrees")
4. distance: COPY THE EXACT NUMERICAL DISTANCE IN INCHES I provided for the region where the object is located

Structure your response as follows in json format:
{
"target": {"name": "___", "region": "___", "angle": "__ degrees", "distance": "__ inches"},
"obstacles": [{"name": "___", "region": "___", "angle": "__ degrees", "distance": "__ inches"}]
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

def add_position_grid(image):
    """Add reference grid with angle markers to image"""
    h, w = image.shape[:2]
    grid_img = image.copy()
    
    # Add horizontal lines
    for i in range(0, h, h//3):
        cv2.line(grid_img, (0, i), (w, i), (255, 255, 255), 1)
    
    # Add vertical lines and angle markers
    angles = [-30, -10, 10, 30]
    for i, angle in enumerate(angles):
        x = w * i // (len(angles) - 1)
        cv2.line(grid_img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(grid_img, f"{angle}°", (x-15, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return grid_img

# Add a helper function to get distance from depth value
def get_distance_from_depth(depth_value):
    """Convert normalized depth value (0-1) to distance in inches
    0 = farthest (approx 10+ feet)
    1 = closest (approx 1 foot)
    """
    # Linear mapping from depth_value (0-1) to distance (120-10 inches)
    estimated_distance = int(72 - (depth_value * 60))
    return estimated_distance

def analyze_depth_regions(depth_map, depth_colormap, num_regions=9):
    """Analyze depth map divided into regions (3x3 grid)"""
    h, w = depth_map.shape[:2]
    region_h, region_w = h // 3, w // 3
    
    regions = []
    for y in range(3):
        for x in range(3):
            y1, y2 = y * region_h, (y + 1) * region_h
            x1, x2 = x * region_w, (x + 1) * region_w
            
            region = depth_map[y1:y2, x1:x2]
            avg_depth = float(np.mean(region))
            
            # Determine position description
            vertical = ["bottom", "middle", "top"][2-y]  # Invert y for intuitive naming
            horizontal = ["left", "center", "right"][x]
            position = f"{vertical}-{horizontal}"
            
            # Calculate center position in degrees (-30 to +30 degrees)
            center_x = (x1 + x2) / 2
            angle = ((center_x / w) - 0.5) * 60  # Map to [-30, 30] degrees
            
            # Get distance directly from depth value
            estimated_distance = get_distance_from_depth(avg_depth)
            
            regions.append({
                "position": position,
                "depth_value": avg_depth,
                "angle": angle,
                "bounds": (x1, y1, x2, y2),
                "distance": estimated_distance
            })
    
    return regions

def process_depth(frame):
    """Process frame through depth model and save both images"""
    if depth_model is None:
        print("! Depth model not available")
        return None, None, frame, frame, []
    
    print("  Processing depth...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    image_tensor = transform(pil_image)
    
    prediction = depth_model.infer(image_tensor)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    
    # Normalize depth for visualization
    inverse_depth = 1 / depth
    max_invdepth = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth = max(1 / 250, inverse_depth.min())
    depth_normalized = (inverse_depth - min_invdepth) / (max_invdepth - min_invdepth)
    
    # Convert to colormap
    depth_colormap = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )
    
    # Add positional analysis with distance information
    depth_regions = analyze_depth_regions(depth_normalized, depth_colormap)
    
    # Create grid overlay versions
    frame_with_grid = add_position_grid(frame)
    depth_map_with_grid = add_position_grid(depth_colormap)
    
    # Add distance info to depth map
    for region in depth_regions:
        x1, y1, x2, y2 = region["bounds"]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Add small marker dot
        cv2.circle(depth_map_with_grid, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Add text with distance in inches
        distance_text = f"{region['distance']} in"
        cv2.putText(depth_map_with_grid, distance_text, 
                    (center_x-20, center_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return depth_colormap, depth_normalized, frame_with_grid, depth_map_with_grid, depth_regions

def camera_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event): # type: ignore
    print("Starting camera process...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("! Error: Could not open camera")
            stop_event.set()
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Camera initialized at 640x480")
    except Exception as e:
        print(f"! Camera error: {str(e)}")
        stop_event.set()
        return
    
    show_menu = False
    show_depth = False
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        display_frame = frame.copy()

        if processing_event.is_set():
            cv2.putText(display_frame, "Processing...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif show_menu:
            display_frame = show_goal_menu(frame)
        else:
            cv2.putText(display_frame, "Press SPACE to select goal, D for depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show_depth:
            depth_map = process_depth(frame)
            combined = np.hstack((display_frame, depth_map))
            cv2.imshow('Camera Feed with Depth', combined)
        else:
            cv2.imshow('Camera Feed', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and not processing_event.is_set():  # Spacebar
            show_menu = True
        elif key == ord('d'):  # Toggle depth view
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow('Camera Feed with Depth')
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
    
    print("Camera process stopped.")
    cap.release()
    cv2.destroyAllWindows()

def inference_process(frame_queue: Queue, trigger_event: Event, processing_event: Event, stop_event: Event): # type: ignore
    print("Starting inference process...")
    llm_loaded = False
    
    while not stop_event.is_set():
        if trigger_event.is_set() and not frame_queue.empty():
            frame, goal = frame_queue.get()
            
            print(f"\n--- Processing new frame with goal: {goal} ---")
            
            # Process images with grid overlays
            print("1. Generating depth map...")
            depth_map, _, frame_with_grid, depth_with_grid, depth_regions = process_depth(frame)
            
            # Save both images with grid overlays
            print("2. Saving processed images...")
            cv2.imwrite('temp_frame.jpg', frame_with_grid)
            if depth_with_grid is not None:
                cv2.imwrite('temp_depth.jpg', depth_with_grid)
            
            # Create depth region summary with enhanced distance information
            print("3. Analyzing depth regions...")
            region_info = "\n\nDepth Analysis by Region (USE THESE EXACT DISTANCES):\n"
            for region in depth_regions:
                region_info += f"• Region {region['position']}: DISTANCE = {region['distance']} inches, angle = {region['angle']:.1f}°\n"
            
            # Add a reminder
            region_info += "\nIMPORTANT: For any object, copy the EXACT distance value from its region.\n"
            
            print("4. Querying LLM...")
            if not llm_loaded:
                print("   (First LLM query may take longer to load the model)")
                llm_loaded = True
                
            try:
                if depth_with_grid is not None:
                    response = ollama.chat(
                        model='llava',
                        messages=[{
                            'role': 'user',
                            'content': "Context: " + CONTEXT + region_info + "\n\nGoal: " + goal + "\nI'm providing two images with grid overlays: the regular view and its depth map.",
                            'images': ['temp_frame.jpg', 'temp_depth.jpg']
                        }]
                    )
                else:
                    response = ollama.chat(
                        model='llava',
                        messages=[{
                            'role': 'user',
                            'content': "Context: " + CONTEXT + "\n\nGoal: " + goal,
                            'images': ['temp_frame.jpg']
                        }]
                    )
                    
                llm_output = response['message']['content']
                print("✓ LLM response received")
                print(f"\nLLM Response: {llm_output}")

                print("5. Generating robot commands...")
                interpreter = RobotCommandInterpreter(goal)
                commands = interpreter.interpret_json(llm_output)
                print("Robot Commands:", commands)
                
            except Exception as e:
                print(f"! Error during inference: {str(e)}")

            processing_event.clear()
            trigger_event.clear()
            print("--- Processing complete ---\n")
    
    print("Inference process stopped.")

def main():
    print("\n=== RoverML System Initializing ===")
    print("Setting up multiprocessing...")
    multiprocessing.set_start_method('spawn')
    frame_queue = Queue(maxsize=1)
    trigger_event = Event()
    processing_event = Event()
    stop_event = Event()
    
    print("Starting processes...")
    processes = [
        Process(target=camera_process, args=(frame_queue, trigger_event, processing_event, stop_event)),
        Process(target=inference_process, args=(frame_queue, trigger_event, processing_event, stop_event))
    ]
    
    for p in processes:
        p.start()
    
    print("\n=== System Ready ===")
    print("Press SPACE to select a goal")
    print("Press 'D' to toggle depth view")
    print("Press 'Q' to quit")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        stop_event.set()
        for p in processes:
            p.terminate()
        print("Processes terminated")
    
    print("=== System Shutdown Complete ===")

if __name__ == '__main__':
    main()