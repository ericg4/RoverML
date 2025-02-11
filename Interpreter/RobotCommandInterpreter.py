import re
import ollama

class RobotCommandInterpreter:
    def __init__(self, goal):
        self.goal = goal
        self.distance_map = {  # fuzzy distance terms
            "close by": 12,  # inches
            "near": 24,      # inches
            "far": 60        # inches
        }
        self.angle_map = {  # fuzzy angle terms
            "slightly left": -15, # degrees
            "slightly right": 15,  # degrees
            "directly ahead": 0
        }


    def parse_distance(self, text):
      """Extracts distance in inches.  Handles numbers + units and fuzzy terms."""
      match = re.search(r"(\d+)\s*(inches|in)", text) #regex to find numbers followed by inches or in
      if match:
          return int(match.group(1))

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

        # 3. Goal-Oriented Actions (Example)
        if self.goal == "Drive the rover to the box":
            if target_object == "box":
                if distance is None:
                    commands.append("Object Detected but Distance Unknown - STOP")
                    return commands
                if angle is None:
                    commands.append("Object Detected but Angle Unknown - STOP")
                    return commands

                if distance > 12:
                    commands.append(f"Turn {angle} degrees")
                    commands.append(f"Forward {distance - 12} inches")  # Approach
                else:
                    commands.append("Approach Target")
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