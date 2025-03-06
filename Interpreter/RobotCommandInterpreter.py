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


    def parse_distance(self, text):
        """Extracts distance in inches. Now handles depth-based descriptions."""
        # First try numerical values
        match = re.search(r"(\d+)\s*(inches|in)", text)
        if match:
            return int(match.group(1))

        # Check depth-based descriptions
        if "very blue" in text.lower() or "darkest blue" in text.lower():
            return self.distance_map["very far"]
        elif "blue" in text.lower():
            return self.distance_map["far"]
        elif "green" in text.lower():
            return self.distance_map["medium"]
        elif "yellow" in text.lower():
            return self.distance_map["near"]
        elif "orange" in text.lower():
            return self.distance_map["close by"]
        elif "red" in text.lower():
            return self.distance_map["very close"]

        # Check standard fuzzy terms
        if text in self.distance_map:
            return self.distance_map[text]

        return None

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

        # 3. Goal-Oriented Actions (Example)
        if self.goal == "Drive the rover to the box":
            if target_object == "box":
                if distance is None:
                    commands.append("Object Detected but Distance Unknown - STOP")
                    return commands
                elif angle is None:
                    commands.append("Object Detected but Angle Unknown - STOP")
                    return commands

                if distance > 12:
                    if (angle > 15):
                        commands.append(f"Turn {angle} degrees right")
                    elif (angle < -15):
                        commands.append(f"Turn {angle} degrees left")
                    commands.append(f"Forward {distance - 12} inches")  # Approach
                else:
                    commands.append(f"Drive forward {distance/2} inches")  # Close enough
                    commands.append("Stop") #At the target, stop
            elif target_object == "obstacle":
                commands.append("Obstacle Detected - STOP") #If obstacle is detected just stop

            else:
                commands.append("Target not found - Scan")
        else:
            commands.append("Unknown goal.  Cannot interpret.")

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
        
        # Fix common formatting issues
        text = re.sub(r'(\w+):', r'"\1":', text)  # Add quotes around keys
        text = text.replace('None', 'null')        # Replace Python None with JSON null
        text = text.replace('True', 'true')        # Replace Python True with JSON true
        text = text.replace('False', 'false')      # Replace Python False with JSON false
        
        return text

    def interpret_json(self, llm_output):
        """Interprets JSON-formatted LLM output and generates robot commands."""
        commands = []
        
        try:
            # Clean and parse JSON response
            cleaned_json = self._prepare_json_text(llm_output)
            print("Cleaned JSON:", cleaned_json)
            data = json.loads(cleaned_json)
            
            # Handle target object
            if "target" in data and data["target"]:
                target = data["target"]
                target_distance = self.parse_distance(target.get("distance", "unknown"))
                target_angle = self.parse_angle(target.get("angle", "unknown"))
                target_name = target.get("name", "unknown")
                
                # Check if this target matches our goal
                goal_target = self.goal.split("to the ")[-1].lower()
                if goal_target in target_name.lower():
                    if target_distance is None or target_angle is None:
                        commands.append("Target detected but position unclear - STOP")
                    else:
                        if target_distance > 12:
                            if abs(target_angle) > 15:
                                commands.append(f"Turn {'right' if target_angle > 0 else 'left'} {abs(target_angle)} degrees")
                            commands.append(f"Forward {target_distance - 12} inches")
                        else:
                            commands.append(f"Approach forward about {target_distance} inches")
                            commands.append(f"Stop - {target_name} reached")
                else:
                    commands.append(f"Found {target_name} but looking for {goal_target} - Continue scanning")

            # Handle obstacles
            if "obstacles" in data:
                for obstacle in data["obstacles"]:
                    obstacle_distance = self.parse_distance(obstacle.get("distance", "unknown"))
                    obstacle_angle = self.parse_angle(obstacle.get("angle", "unknown"))
                    
                    if obstacle_distance and obstacle_distance < 24:  # If obstacle is within 2 feet
                        commands.append(f"Warning: Obstacle at {obstacle_distance} inches, {obstacle_angle} degrees")
                        if abs(obstacle_angle) < 30:  # If obstacle is roughly ahead
                            commands.append("Stop - Obstacle in path")
                            return commands  # Priority to avoid obstacles

        except json.JSONDecodeError:
            commands.append("Error: Could not parse LLM response")
        except Exception as e:
            commands.append(f"Error: {str(e)}")

        return commands if commands else ["No actionable commands - Continue scanning"]