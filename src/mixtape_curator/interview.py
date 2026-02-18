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
        
        # Spec v2.1.1 Confidence Axes
        self.confidence = {
            "constraints": 0.0,
            "intent": 0.0,
            "cohesion": 0.0,
            "energy_mood": 0.0
        }
        
    def start(self) -> str:
        """Begin the 5-stage interview process."""
        self.history = []
        return "Hi! I'm your AI Mixtape Curator. Who am I making this mix for today? (e.g. 'for me', 'for a road trip')"

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process user input through 5 Spec Stages.
        Returns: (next_question_text, is_complete)
        """
        self.history.append({"role": "user", "content": user_input})
        q_count = len([m for m in self.history if m["role"] == "assistant"])
        
        # --- STAGE 1: Intake (Recipient, Context, Constraints) ---
        if q_count == 0:
            self.profile.recipient = user_input
            self.confidence["intent"] += 0.2
            reply = "Got it. What's the context or vibe? (e.g. 'late night study', 'high-energy gym', 'chill backyard')"
            self._log_reply(reply)
            return reply, False
            
        if q_count == 1:
            self.profile.context_notes = user_input
            self.confidence["intent"] += 0.2
            reply = "Are there any artists or tracks I MUST include? Or any artists/genres I should strictly EXCLUDE?"
            self._log_reply(reply)
            return reply, False

        if q_count == 2:
            # Intake of Must-includes/Excludes (Naive extraction for demo)
            lower_input = user_input.lower()
            if "exclude" in lower_input or "no " in lower_input or "not " in lower_input:
                # Try to extract artist after the keyword
                for word in ["exclude", "no ", "not ", "dont want ", "don't want "]:
                    if word in lower_input:
                        parts = lower_input.split(word)
                        if len(parts) > 1:
                            artist = parts[1].strip().title()
                            if artist:
                                self.profile.exclude_artists.append(artist)
                                logger.info(f"Adding {artist} to exclusion list.")
            
            if "include" in lower_input or "must have" in lower_input:
                for word in ["include", "must have", "want "]:
                    if word in lower_input:
                        parts = lower_input.split(word)
                        if len(parts) > 1:
                            artist = parts[1].strip().title()
                            if artist:
                                self.profile.must_include_artists.append(artist)
            
            self.confidence["constraints"] = 0.8
            # Next: Stage 2 GenreCalibration
            reply = "Which genres should I focus on? (e.g. '80s synthpop', 'classical', 'hip-hop')"
            self._log_reply(reply)
            return reply, False

        # --- STAGE 2: Genre & Descriptor Calibration ---
        if q_count == 3:
            self.profile.target_genres = [g.strip() for g in user_input.split(',')]
            self.confidence["intent"] += 0.3
            reply = "And what specific textures or moods (moody, bright, aggressive, neon) are we aiming for?"
            self._log_reply(reply)
            return reply, False
            
        if q_count == 4:
            self.profile.target_descriptors = [d.strip() for d in user_input.split(',')]
            self.confidence["intent"] = 1.0 
            # Next: Stage 3 Cohesion
            reply = "Should this mix be highly consistent (Uniform) or vary wildly (Eclectic)?"
            self._log_reply(reply)
            return reply, False
            
        # --- STAGE 3: Cohesion Calibration ---
        if q_count == 5:
            if "uniform" in user_input.lower() or "consistent" in user_input.lower():
                self.profile.targets.uniformity = 0.9
            elif "eclectic" in user_input.lower() or "varied" in user_input.lower():
                self.profile.targets.uniformity = 0.1
            self.confidence["cohesion"] = 1.0
            # Next: Stage 4 Energy/Mood
            reply = "Finally, what's its desired energy level? (Low/Chill, Moderate, or Absolute Maximum?)"
            self._log_reply(reply)
            return reply, False
            
        # --- STAGE 4: Energy/Mood Calibration ---
        if q_count == 6:
            if "high" in user_input.lower() or "maximum" in user_input.lower():
                self.profile.targets.energy = 0.9
                self.profile.targets.intensity = 0.8
            elif "chill" in user_input.lower() or "low" in user_input.lower():
                self.profile.targets.energy = 0.2
                self.profile.targets.intensity = 0.2
            self.confidence["energy_mood"] = 1.0
            
            summary = self._generate_summary()
            reply = f"Calibration complete. Here is the blueprint:\n{summary}\n\nShall I begin the generation? (yes/no)"
            self._log_reply(reply)
            return reply, False

        # --- STAGE 5: Confirmation ---
        if q_count >= 7:
            if "yes" in user_input.lower():
                self.completed = True
                return "Initializing ReAct Agent for playlist construction...", True
            else:
                reply = "What specific axis (Genres, Cohesion, Energy) should I recalibrate?"
                self._log_reply(reply)
                return reply, False
        
        return "ERROR", True

    def _log_reply(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def _generate_summary(self) -> str:
        return (
            f"- For: {self.profile.recipient}\n"
            f"- Context: {self.profile.context_notes}\n"
            f"- Genres/Vibes: {', '.join(self.profile.target_genres)}\n"
            f"- Cohesion: {'Strictly Uniform' if self.profile.targets.uniformity > 0.7 else 'Highly Eclectic' if self.profile.targets.uniformity < 0.3 else 'Balanced'}\n"
            f"- Energy: {self.profile.targets.energy * 100:.0f}%\n"
            f"- Confidence: {sum(self.confidence.values())/4 * 100:.0f}%"
        )

interviewer = Interviewer()
