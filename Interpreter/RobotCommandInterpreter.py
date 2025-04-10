import math
import re
import ollama
import json

class RobotCommandInterpreter:
    def __init__(self, goal):
        self.goal = goal
        self.distance_map = {  # fuzzy distance terms
            "close by": 12,    # inches
            "near": 24,        # inches
            "far": 60,         # inches
            "very close": 6,   # inches
            "medium": 36,      # inches
            "very far": 120    # inches
        }
        self.angle_map = {  # fuzzy angle terms
            "slightly left": -15, # degrees
            "slightly right": 15,  # degrees
            "directly ahead": 0
        }
        
        self.WHEELBASE = 0.3  # meters
        self.DEFAULT_VELOCITY = 0.2  # default linear velocity in m/s

    def inches_to_meters(self, inches):
        """Convert inches to meters"""
        return inches * 0.0254
        
    def degrees_to_radians(self, degrees):
        """Convert degrees to radians"""
        return degrees * math.pi / 180

    def compute_arc_wheel_speeds(self, d, theta, L, v):
        """
        Compute left and right wheel speeds for arc-based motion.

        Parameters:
        - d (float): distance to target (in meters)
        - theta (float): angle to target (in radians, positive = right)
        - L (float): wheelbase width (distance between wheels in meters)
        - v (float): desired linear speed of the robot's center (in m/s)

        Returns:
        - v_l (float): left wheel speed (in m/s)
        - v_r (float): right wheel speed (in m/s)
        - t (float): time to reach the target point (in seconds)
        """
        # Handle straight-line case to avoid division by zero
        if abs(theta) < 1e-6:
            v_l = v_r = v
            t = d / v
            return v_l, v_r, t

        # Radius of the arc to follow
        R = d / (2 * math.sin(theta))

        # Compute individual wheel speeds
        v_l = v * ((R - (L / 2)) / R)
        v_r = v * ((R + (L / 2)) / R)

        # Arc length = R * |theta|, time = arc length / v
        arc_length = abs(R * theta)
        t = arc_length / v

        return v_l, v_r, t


    def parse_distance(self, text):
        """Extracts distance in inches from text"""
        # First try numerical values with units
        match = re.search(r"(\d+)\s*(inches|in)", text)
        if match:
            return int(match.group(1))
        
        # Try just a number (assume inches)
        match = re.search(r"^(\d+)$", text)
        if match:
            return int(match.group(1))
        
        # Check standard fuzzy terms as fallback
        if text in self.distance_map:
            return self.distance_map[text]
        
        return None  # Could not parse distance

    def parse_angle(self, text):
      """Extracts angle in degrees.  Handles numbers + units and fuzzy terms."""
      match = re.search(r"(\d+)\s*(degrees|deg)", text)
      if match:
          angle = int(match.group(1))
          if "left" in text:
              return -angle  # Negative for left
          return angle

      if text in self.angle_map:
          return self.angle_map[text]

      return None

    def parse_object(self, text):
        """Simple object extraction (expand as needed)."""
        if "box" in text.lower():
            return "box"
        elif "obstacle" in text.lower():
            return "obstacle"
        return None

    def interpret(self, llm_output):
        """Main interpretation function."""
        commands = []

        # 1. Object Detection
        target_object = self.parse_object(llm_output)

        # 2. Distance & Angle
        distance = self.parse_distance(llm_output)
        angle = self.parse_angle(llm_output)

        # 3. Goal-Oriented Actions with wheel speed calculations
        if self.goal == "Drive the rover to the box":
            if target_object == "box":
                if distance is None:
                    commands.append("Object Detected but Distance Unknown - STOP")
                    return commands
                elif angle is None:
                    commands.append("Object Detected but Angle Unknown - STOP")
                    return commands

                # Convert units for calculation
                distance_m = self.inches_to_meters(distance)
                angle_rad = self.degrees_to_radians(angle)
                
                # Calculate wheel speeds and time
                v_left, v_right, t = self.compute_arc_wheel_speeds(
                    distance_m, angle_rad, self.WHEELBASE, self.DEFAULT_VELOCITY
                )
                
                commands.append(f"Left wheel: {v_left:.2f} m/s, Right wheel: {v_right:.2f} m/s, Time: {t:.2f} s")
                
                if distance <= 12:
                    commands.append("Stop - Target reached")  # At the target, stop
            
            elif target_object == "obstacle":
                commands.append("Obstacle Detected - STOP") # If obstacle is detected just stop

            else:
                commands.append("Target not found - Scan")
        else:
            commands.append("Unknown goal. Cannot interpret.")

        return commands

    def interpret_with_llm(self, llm_output):
        """An example using LLM to decide which command to follow"""
        response = ollama.chat(
            model='deepseek-r1:7b',
            messages=[{
                'role': 'user',
                'content': f"""Given that you are an ai in an autonomous rover and you are trying to reach the goal: {self.goal}.  You have the following descriptions of objects: {llm_output}.

                Choose the best course of action from the following list of commands:
                'Forward 10 inches', 'Turn 90 degrees left', 'Stop', 'Scan', 'Avoid Object', 'Grasp'.

                Return the command that best describes the best action.
                """,
            }]
        )
        return [response['message']['content']]

    def _prepare_json_text(self, text):
        """Clean and prepare raw text for JSON parsing"""
        # Remove any markdown code block markers
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # Find the first '{' and last '}' to extract just the JSON part
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        
        # Handle degree symbols and format angle values properly
        # First, fix the degree symbol
        text = re.sub(r'(-?\d+\.\d+)°', r'\1 degrees', text)  # Convert -0.0° to "-0.0 degrees"
        text = re.sub(r'(-?\d+)°', r'\1 degrees', text)       # Convert -10° to "-10 degrees"
        
        # Fix common formatting issues
        text = re.sub(r'(\w+):', r'"\1":', text)  # Add quotes around keys
        text = text.replace('None', 'null')        # Replace Python None with JSON null
        text = text.replace('True', 'true')        # Replace Python True with JSON true
        text = text.replace('False', 'false')      # Replace Python False with JSON false
        
        # Fix quotes around values that already have quotes
        text = re.sub(r'""-?(\d+(?:\.\d+)?)\s+degrees""', r'"\1 degrees"', text)  # Fix double quotes
        text = re.sub(r'""([^"]+)""', r'"\1"', text)  # Fix any other double quoted values
        
        # Replace placeholder values
        text = text.replace('"USE THE PRE-COMPUTED VALUE"', '"0 inches"')  # Replace placeholders
                
        # Validate JSON structure before returning
        try:
            # Simple test to see if it parses
            json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON may still have issues: {str(e)}")
            print(f"Current JSON string: {text}")
        
        return text

    def interpret_json(self, llm_output):
        """Interprets JSON-formatted LLM output with enhanced positional awareness."""
        commands = []
        
        try:
            # Clean and parse JSON response
            cleaned_json = self._prepare_json_text(llm_output)
            print("Cleaned JSON:", cleaned_json)
            data = json.loads(cleaned_json)
            
            # Handle target object
            if "target" in data and data["target"]:
                target = data["target"]
                target_name = target.get("name", "unknown")
                target_region = target.get("region", "unknown")
                target_distance = self.parse_distance(target.get("distance", "unknown"))
                target_angle = self.parse_angle(target.get("angle", "unknown"))
                
                # Print enhanced information
                print(f"Target: {target_name} in {target_region} region")
                
                # Check if this target matches our goal
                goal_target = self.goal.split("to the ")[-1].lower()
                if goal_target in target_name.lower():
                    if target_distance is None or target_angle is None:
                        commands.append("Target detected but position unclear - STOP")
                    else:
                        # Convert units for calculation
                        distance_m = self.inches_to_meters(target_distance)
                        angle_rad = self.degrees_to_radians(target_angle)
                        
                        # Calculate wheel speeds and time
                        v_left, v_right, t = self.compute_arc_wheel_speeds(
                            distance_m, angle_rad, self.WHEELBASE, self.DEFAULT_VELOCITY
                        )
                        
                        commands.append(f"Left wheel: {v_left:.2f} m/s, Right wheel: {v_right:.2f} m/s, Time: {t:.2f} s")
                        
                        if target_distance <= 12:
                            commands.append(f"Stop - {target_name} reached in {target_region} region")
                else:
                    commands.append(f"Found {target_name} but looking for {goal_target} - Continue scanning")

            # Handle obstacles with positional awareness
            if "obstacles" in data:
                for obstacle in data["obstacles"]:
                    obstacle_distance = self.parse_distance(obstacle.get("distance", "unknown"))
                    obstacle_angle = self.parse_angle(obstacle.get("angle", "unknown"))
                    obstacle_region = obstacle.get("region", "unknown")
                    
                    if obstacle_distance and obstacle_distance < 24:  # If obstacle is within 2 feet
                        commands.append(f"Warning: {obstacle.get('name', 'Obstacle')} at {obstacle_distance} inches, {obstacle_angle} degrees in {obstacle_region}")
                        commands.append("Stop - Obstacle in path")
                        return commands  # Priority to avoid obstacles

        except json.JSONDecodeError:
            commands.append("Error: Could not parse LLM response")
        except Exception as e:
            commands.append(f"Error: {str(e)}")

        return commands if commands else ["No actionable commands - Continue scanning"]