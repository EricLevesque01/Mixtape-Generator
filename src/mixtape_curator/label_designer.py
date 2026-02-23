"""
CD Label Designer — generates printable CD jewel case labels as PDF.
Supports user-uploaded images or AI-generated cover art via DALL-E.
Standard jewel case insert: 4.75" × 4.75" (120.65mm × 120.65mm).
"""

import os
import logging
from pathlib import Path
from typing import Optional
from fpdf import FPDF

from mixtape_curator.library import library
from mixtape_curator.models import Playlist

logger = logging.getLogger("mixtape_curator.label")

LABEL_SIZE_MM = 120.65  # Standard CD jewel case insert
EXPORTS_DIR = Path("exports/labels")


def create_cd_label(
    title: str,
    art_prompt: Optional[str] = None,
    playlist: Optional[Playlist] = None,
    image_path: Optional[str] = None,
) -> str:
    """
    Generate a CD label PDF.
    
    Args:
        title: Mixtape title
        art_prompt: Text prompt for AI image generation (uses DALL-E)
        playlist: Playlist object to include tracklist
        image_path: Path to user-uploaded cover art
    
    Returns:
        Path to generated PDF file.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate cover art if prompt provided and no image
    if art_prompt and not image_path:
        image_path = _generate_cover_art(art_prompt, title)
    
    # Create PDF
    pdf = FPDF(unit="mm")
    pdf.set_auto_page_break(auto=False)
    
    # ── Front Cover (Page 1) ───────────────────────────────────────
    pdf.add_page()
    _draw_front_cover(pdf, title, image_path)
    
    # ── Back Cover / Tracklist (Page 2) ────────────────────────────
    if playlist:
        pdf.add_page()
        _draw_back_cover(pdf, title, playlist)
    
    # Save
    safe_title = title.replace(" ", "_").replace("/", "-")[:30]
    pdf_path = EXPORTS_DIR / f"{safe_title}_label.pdf"
    pdf.output(str(pdf_path))
    
    logger.info(f"CD label saved to {pdf_path}")
    return str(pdf_path)


def _draw_front_cover(pdf: FPDF, title: str, image_path: Optional[str]):
    """Draw the front cover page."""
    # Center the label area
    margin_x = (210 - LABEL_SIZE_MM) / 2  # A4 width = 210mm
    margin_y = (297 - LABEL_SIZE_MM) / 2  # A4 height = 297mm
    
    # Draw cut guide border
    pdf.set_draw_color(180, 180, 180)
    pdf.set_dash_pattern(dash=2, gap=2)
    pdf.rect(margin_x, margin_y, LABEL_SIZE_MM, LABEL_SIZE_MM)
    pdf.set_dash_pattern()
    
    if image_path and Path(image_path).exists():
        # Scale image to fit label area with 2mm padding
        pad = 2
        pdf.image(
            image_path,
            x=margin_x + pad,
            y=margin_y + pad,
            w=LABEL_SIZE_MM - 2 * pad,
            h=LABEL_SIZE_MM - 2 * pad,
        )
    else:
        # No image — draw a styled gradient-look background
        pdf.set_fill_color(30, 30, 40)
        pdf.rect(margin_x, margin_y, LABEL_SIZE_MM, LABEL_SIZE_MM, "F")
    
    # Title overlay at bottom
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    title_y = margin_y + LABEL_SIZE_MM - 20
    pdf.set_xy(margin_x + 5, title_y)
    pdf.cell(LABEL_SIZE_MM - 10, 10, title, align="C")
    
    # Subtitle
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_xy(margin_x + 5, title_y + 10)
    pdf.cell(LABEL_SIZE_MM - 10, 5, "Curated by AI Mixtape Curator", align="C")


def _draw_back_cover(pdf: FPDF, title: str, playlist: Playlist):
    """Draw the back cover with tracklist."""
    margin_x = (210 - LABEL_SIZE_MM) / 2
    margin_y = (297 - LABEL_SIZE_MM) / 2
    
    # Cut guide
    pdf.set_draw_color(180, 180, 180)
    pdf.set_dash_pattern(dash=2, gap=2)
    pdf.rect(margin_x, margin_y, LABEL_SIZE_MM, LABEL_SIZE_MM)
    pdf.set_dash_pattern()
    
    # Background
    pdf.set_fill_color(20, 20, 30)
    pdf.rect(margin_x, margin_y, LABEL_SIZE_MM, LABEL_SIZE_MM, "F")
    
    # Title at top
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(margin_x + 5, margin_y + 5)
    pdf.cell(LABEL_SIZE_MM - 10, 8, title, align="C")
    
    # Divider line
    pdf.set_draw_color(100, 100, 120)
    pdf.line(margin_x + 10, margin_y + 16, margin_x + LABEL_SIZE_MM - 10, margin_y + 16)
    
    # Tracklist
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(220, 220, 230)
    
    y = margin_y + 20
    max_tracks = min(len(playlist.track_ids), 28)  # Cap for space
    
    for i, tid in enumerate(playlist.track_ids[:max_tracks]):
        t = library.get_track(tid)
        if not t:
            continue
        
        dur_min = t.duration_s // 60
        dur_sec = t.duration_s % 60
        
        # Track number
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_xy(margin_x + 5, y)
        pdf.cell(8, 3.5, f"{i+1}.", align="R")
        
        # Artist - Title
        pdf.set_font("Helvetica", "", 7)
        pdf.set_xy(margin_x + 14, y)
        text = f"{t.artist} — {t.title}"
        if len(text) > 40:
            text = text[:38] + "…"
        pdf.cell(80, 3.5, text)
        
        # Duration
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(150, 150, 160)
        pdf.set_xy(margin_x + LABEL_SIZE_MM - 20, y)
        pdf.cell(15, 3.5, f"{dur_min}:{dur_sec:02d}", align="R")
        pdf.set_text_color(220, 220, 230)
        
        y += 3.8
    
    # Footer with total duration and score
    total_min = playlist.total_duration_s // 60
    total_sec = playlist.total_duration_s % 60
    
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(120, 120, 140)
    footer_y = margin_y + LABEL_SIZE_MM - 8
    pdf.set_xy(margin_x + 5, footer_y)
    pdf.cell(
        LABEL_SIZE_MM - 10, 4,
        f"{len(playlist.track_ids)} tracks  ·  {total_min}:{total_sec:02d}  ·  Score: {playlist.scores.total:.2f}",
        align="C"
    )


def _generate_cover_art(prompt: str, title: str) -> Optional[str]:
    """Generate cover art using DALL-E (OpenAI)."""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY set, skipping cover art generation")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        full_prompt = (
            f"CD album cover art for a mixtape called '{title}'. "
            f"Style: {prompt}. "
            "Square format, no text or typography, artistic and visually striking."
        )
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        
        # Download and save
        import urllib.request
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        img_path = EXPORTS_DIR / f"{title.replace(' ', '_')}_cover.png"
        urllib.request.urlretrieve(image_url, str(img_path))
        
        logger.info(f"Generated cover art: {img_path}")
        return str(img_path)
        
    except Exception as e:
        logger.error(f"Cover art generation failed: {e}")
        return None
