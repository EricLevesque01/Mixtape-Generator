import logging
import json
from typing import List, Dict, Tuple, Optional
from .models import UserProfile, FeedbackTargets
from .config import config

logger = logging.getLogger("mixtape_curator")

class Interviewer:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.profile = UserProfile()
        self.completed = False
        self.max_questions = config.get("max_questions", 10)
        
    def start(self) -> str:
        """Begin the interview process."""
        self.history = []
        return "Hi! Who am I making a mixtape for today? (e.g. 'for me', 'for my partner', 'party mix')"

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process user input, update profile, return next question.
        Returns: (next_question_text, is_complete)
        """
        self.history.append({"role": "user", "content": user_input})
        
        # In a real implementation with LLM, we would classify the input 
        # and update the structured profile.
        # For this CLI draft, we will simulate the "Stages" from Spec v2.1.1 
        # using a simple state machine based on history length.
        
        q_count = len([m for m in self.history if m["role"] == "assistant"])
        
        # Stage 1: Intake (Recipient/Context) - Handled by start()
        if q_count == 0:
            # Assume first answer was recipient
            self.profile.recipient = user_input
            # Next: Stage 2 Genre/Descriptor
            reply = "Got it. What genres or specific vibes should I hunt for? (e.g. '80s synthpop', 'dark techno')"
            self._log_reply(reply)
            return reply, False
            
        # Stage 2: Genre & Descriptor Calibration
        if q_count == 1:
            # Naive parsing for demo
            self.profile.target_genres = [g.strip() for g in user_input.split(',')]
            # Raise genre strictness since user explicitly named genres
            self.profile.genre_strictness = 0.65
            # Next: Stage 3 Cohesion Calibration (Uniform vs Eclectic)
            reply = "Understood. Should this mix feel consistent and uniform, or more eclectic and varied?"
            self._log_reply(reply)
            return reply, False
            
        # Stage 3: Cohesion Calibration
        if q_count == 2:
            if "uniform" in user_input.lower() or "consistent" in user_input.lower():
                self.profile.targets.uniformity = 0.8
            elif "eclectic" in user_input.lower() or "varied" in user_input.lower():
                self.profile.targets.uniformity = 0.2
            else:
                self.profile.targets.uniformity = 0.5 # Default balanced
                
            # Next: Stage 4 Energy/Mood
            reply = "Okay. How about energy? High energy for a workout, or chill constraints?"
            self._log_reply(reply)
            return reply, False
            
        # Stage 4: Energy & Mood
        if q_count == 3:
            if "high" in user_input.lower():
                self.profile.targets.energy = 0.8
                self.profile.targets.intensity = 0.7
            elif "chill" in user_input.lower():
                self.profile.targets.energy = 0.3
                self.profile.targets.intensity = 0.3
            
            # Next: Stage 5 Confirmation
            summary = self._generate_summary()
            reply = f"Great. Here is the plan:\n{summary}\n\nReady to generate? (yes/no)"
            self._log_reply(reply)
            return reply, False
            
        # Stage 5: Confirmation
        if q_count >= 4:
            if "yes" in user_input.lower():
                self.completed = True
                return "Generating your mixtape...", True
            else:
                reply = "What would you like to change?"
                self._log_reply(reply)
                return reply, False
        
        return "ERROR", True

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def _generate_summary(self) -> str:
        return f"- Recipient: {self.profile.recipient}\n- Genres: {self.profile.target_genres}\n- Cohesion: {'Uniform' if self.profile.targets.uniformity > 0.6 else 'Eclectic' if self.profile.targets.uniformity < 0.4 else 'Balanced'}\n- Energy Target: {self.profile.targets.energy}"

interviewer = Interviewer()
