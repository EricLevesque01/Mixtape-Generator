import logging
import json
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
from .models import UserProfile, FeedbackTargets
from .config import config

logger = logging.getLogger("mixtape_curator")

class InterviewState(Enum):
    INTAKE_RECIPIENT = auto()
    INTAKE_CONTEXT = auto()
    INTAKE_CONSTRAINTS = auto()
    CALIBRATION_GENRE = auto()
    CALIBRATION_DESCRIPTOR = auto()
    CALIBRATION_COHESION = auto()
    CALIBRATION_ENERGY = auto()
    CONFIRMATION = auto()
    COMPLETED = auto()

class Interviewer:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.profile = UserProfile()
        self.completed = False
        self.max_questions = config.get("max_questions", 10)
        self.state = InterviewState.INTAKE_RECIPIENT
        
        # Spec v2.1.1 Confidence Axes
        self.confidence = {
            "constraints": 0.0,
            "intent": 0.0,
            "cohesion": 0.0,
            "energy_mood": 0.0
        }
        
    def start(self) -> str:
        """Begin the dynamic interview process."""
        self.history = []
        self.state = InterviewState.INTAKE_RECIPIENT
        return "Hi! I'm your AI Mixtape Curator. Who am I making this mix for today? (e.g. 'for me', 'for a road trip')"

    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process user input through Dynamic State Machine.
        Returns: (next_question_text, is_complete)
        """
        self.history.append({"role": "user", "content": user_input})
        
        # Global Entity Extraction (Run every turn)
        self._parse_constraints(user_input)
        
        # --- STATE MACHINE ---
        
        if self.state == InterviewState.INTAKE_RECIPIENT:
            self.profile.recipient = user_input
            self.confidence["intent"] += 0.2
            
            # Dynamic Skip: If input mentions context ("workout", "party", "study")
            context_keywords = ["workout", "party", "study", "drive", "focus", "sleep", "gym", "run"]
            if any(k in user_input.lower() for k in context_keywords):
                self.profile.context_notes = user_input # Infer context from recipient answer
                self.confidence["intent"] += 0.3
                self.state = InterviewState.INTAKE_CONSTRAINTS # Skip Context Question
                reply = "Got it. Are there any artists or tracks I MUST include? Or any artists/genres I should strictly EXCLUDE?"
                self._log_reply(reply)
                return reply, False
            
            self.state = InterviewState.INTAKE_CONTEXT
            reply = "Got it. What's the context or vibe? (e.g. 'late night study', 'high-energy gym', 'chill backyard')"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.INTAKE_CONTEXT:
            self.profile.context_notes = user_input
            self.confidence["intent"] += 0.2
            self.state = InterviewState.INTAKE_CONSTRAINTS
            reply = "Are there any artists or tracks I MUST include? Or any artists/genres I should strictly EXCLUDE?"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.INTAKE_CONSTRAINTS:
            # Constraints parsed globally now, but we still use this state to advance
            self.confidence["constraints"] = 0.8
            self.state = InterviewState.CALIBRATION_GENRE
            reply = "Which genres should I focus on? (e.g. '80s synthpop', 'classical', 'hip-hop')"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.CALIBRATION_GENRE:
            self.profile.target_genres = [g.strip() for g in user_input.split(',')]
            self.confidence["intent"] += 0.3
            self.state = InterviewState.CALIBRATION_DESCRIPTOR
            reply = "And what specific textures or moods (moody, bright, aggressive, neon) are we aiming for?"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.CALIBRATION_DESCRIPTOR:
            self.profile.target_descriptors = [d.strip() for d in user_input.split(',')]
            self.confidence["intent"] = 1.0 
            self.state = InterviewState.CALIBRATION_COHESION
            reply = "Should this mix be highly consistent (Uniform) or vary wildly (Eclectic)?"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.CALIBRATION_COHESION:
            # Check for vague answer
            vague_keywords = ["idk", "unsure", "whatever", "doesn't matter", "not sure"]
            if any(k in user_input.lower() for k in vague_keywords):
                self.confidence["cohesion"] = 0.2
                # Stay in this state, ask clarifying question
                reply = "To clarify: Do you want a smooth flow where songs sound similar (Uniform), or a journey through different styles (Eclectic)?"
                self._log_reply(reply)
                return reply, False

            if "uniform" in user_input.lower() or "consistent" in user_input.lower() or "smooth" in user_input.lower():
                self.profile.targets.uniformity = 0.9
            elif "eclectic" in user_input.lower() or "varied" in user_input.lower() or "journey" in user_input.lower():
                self.profile.targets.uniformity = 0.1
            self.confidence["cohesion"] = 1.0
            self.state = InterviewState.CALIBRATION_ENERGY
            reply = "Finally, what's its desired energy level? (Low/Chill, Moderate, or Absolute Maximum?)"
            self._log_reply(reply)
            return reply, False
            
        elif self.state == InterviewState.CALIBRATION_ENERGY:
            if "high" in user_input.lower() or "maximum" in user_input.lower():
                self.profile.targets.energy = 0.9
                self.profile.targets.intensity = 0.8
            elif "chill" in user_input.lower() or "low" in user_input.lower():
                self.profile.targets.energy = 0.2
                self.profile.targets.intensity = 0.2
            else:
                 # Moderate default
                 self.profile.targets.energy = 0.5
                 self.profile.targets.intensity = 0.5
                 
            self.confidence["energy_mood"] = 1.0
            self.state = InterviewState.CONFIRMATION
            
            summary = self._generate_summary()
            reply = f"Calibration complete. Here is the blueprint:\n{summary}\n\nShall I begin the generation? (yes/no)"
            self._log_reply(reply)
            return reply, False

        elif self.state == InterviewState.CONFIRMATION:
            if "yes" in user_input.lower():
                self.completed = True
                self.state = InterviewState.COMPLETED
                return "Initializing ReAct Agent for playlist construction...", True
            elif "no" in user_input.lower():
                # Loop back logic not fully implemented, just ask what to change
                reply = "What specific axis (Genres, Cohesion, Energy) should I recalibrate?"
                self._log_reply(reply)
                return reply, False
            else:
                # Handle recalibration inputs or re-prompt
                if "energy" in user_input.lower():
                    self.state = InterviewState.CALIBRATION_ENERGY
                    return "Okay, let's reset Energy. What is the desired energy level?", False
                if "genre" in user_input.lower():
                    self.state = InterviewState.CALIBRATION_GENRE
                    return "Okay, let's reset Genres. Which genres?", False
                
                reply = "Please type 'yes' to start Generation, or 'no' to adjust."
                self._log_reply(reply)
                return reply, False
        
        return "ERROR: Unknown State", True

    def _parse_constraints(self, user_input: str):
        """Naive extraction of constraints."""
        lower_input = user_input.lower()
        if "exclude" in lower_input or "no " in lower_input or "not " in lower_input:
            for word in ["exclude", "no ", "not ", "dont want ", "don't want "]:
                if word in lower_input:
                    parts = lower_input.split(word)
                    if len(parts) > 1:
                        # Split by comma or simple space if just one name
                        raw = parts[1].split(',')[0].strip()
                        artist = raw.title() 
                        if artist:
                            self.profile.exclude_artists.append(artist)
                            logger.info(f"Adding {artist} to exclusion list.")
        
        if "include" in lower_input or "must have" in lower_input:
            for word in ["include", "must have", "want "]:
                if word in lower_input:
                    parts = lower_input.split(word)
                    if len(parts) > 1:
                        raw = parts[1].split(',')[0].strip()
                        artist = raw.title()
                        if artist:
                            self.profile.must_include_artists.append(artist)

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
        
    def refine_profile(self, feedback: str):
        """
        Adjust profile based on user feedback during Neither loop.
        Simple keyword-based logic for now.
        """
        feedback = feedback.lower()
        
        # Energy adjustments
        if any(w in feedback for w in ["slow", "sleepy", "boring", "low energy", "faster"]):
            self.profile.targets.energy = min(1.0, self.profile.targets.energy + 0.2)
            logger.info("Increasing target energy.")
            
        if any(w in feedback for w in ["fast", "intense", "aggressive", "too hard", "slower"]):
            self.profile.targets.energy = max(0.0, self.profile.targets.energy - 0.2)
            logger.info("Decreasing target energy.")
            
        # Cohesion adjustments
        if any(w in feedback for w in ["messy", "random", "all over", "inconsistent", "too eclectic"]):
            self.profile.targets.uniformity = min(1.0, self.profile.targets.uniformity + 0.2)
            logger.info("Increasing uniformity.")
            
        if any(w in feedback for w in ["samey", "repetitive", "boring", "too similar", "vary"]):
            self.profile.targets.uniformity = max(0.0, self.profile.targets.uniformity - 0.2)
            logger.info("Decreasing uniformity.")
            
        # Exclusions (Simple "exclude X" parsing)
        if "exclude" in feedback or "no " in feedback:
            for word in ["exclude", "no ", "not ", "dont want ", "don't want "]:
                if word in feedback:
                    parts = feedback.split(word)
                    if len(parts) > 1:
                        # naive: assume rest of string is artist or comma sep
                        artist_chunk = parts[1].split(',')[0].strip().title()
                        if artist_chunk and artist_chunk not in self.profile.exclude_artists:
                            self.profile.exclude_artists.append(artist_chunk)
                            logger.info(f"Adding '{artist_chunk}' to exclusions.")
                            
interviewer = Interviewer()
