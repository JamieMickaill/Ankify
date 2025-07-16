import os
import sys
import json
import base64
import random
import time
import pickle
import html
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pymupdf as fitz
from PIL import Image
import io
import requests
from datetime import datetime
import re
import genanki
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class PromptTemplates:
    """Centralized prompt templates to avoid duplication."""
    
    @staticmethod
    def get_cloze_instruction(single_card_mode: bool) -> str:
        """Get the appropriate cloze instruction based on card mode."""
        if single_card_mode:
            return """IMPORTANT: Use ONLY {{c1::}} for ALL cloze deletions (this creates a single card with multiple blanks revealed simultaneously)."""
        return """IMPORTANT: Create cloze deletions using {{c1::}}, {{c2::}}, {{c3::}} etc. for different blanks within the same card."""
    
    @staticmethod
    def get_common_rules() -> str:
        """Get common card creation rules used in both analysis and critique."""
        return """
CRITICAL RULES FOR CARD CREATION:
1. Cards will be reviewed WITHOUT the context of the lecture - ensure each card is self-contained
2. Avoid ambiguous cloze deletions - the answer should be clear from the surrounding context
3. Focus on learning outcomes and objectives if shown on the slide
4. DO NOT create cards about the title, learning objectives, or outline themselves

WHAT NOT TO TEST:
❌ "This lecture focuses on {{c1::treatment of hormone-receptor-positive breast cancer}}" (just restates title)
❌ "A key learning objective is to explain {{c1::tamoxifen benefit}}" (tests the objective, not the content)
❌ "Today we will discuss {{c1::three types of breast cancer}}" (tests outline, not facts)

Instead, use learning objectives to GUIDE what medical facts to extract from content slides."""
    
    @staticmethod
    def get_cloze_examples() -> str:
        """Get good vs bad cloze examples."""
        return """
GOOD vs BAD CLOZE EXAMPLES:
✅ GOOD: "The IVC is formed by the junction of the {{c1::left and right common iliac veins}}"
❌ BAD: "The IVC is {{c1::formed by the junction of the left and right common iliac veins}}" (too vague)

✅ GOOD: "Malignant pericardial effusion should not contain {{c1::malignant}} cells"
❌ BAD: "The fluid shouldn't contain {{c1::malignant}} cells" (which fluid?)

✅ GOOD: "Lung cancer prognosis is poor when {{c1::T cells}} are {{c2::inactivated}}"
❌ BAD: "Lung cancer prognosis is poor when {{c1::T cells are inactivated}}" (tests too much at once)"""
    
    @staticmethod
    def get_advanced_principles() -> str:
        """Get advanced cloze principles."""
        return """
ADVANCED CLOZE PRINCIPLES:

❌ AVOID: "Tamoxifen for {{c1::premenopausal}}, AIs for {{c2::postmenopausal}}" 
→ Problem: c2 makes c1 obvious (binary choice)
✅ BETTER: "Tamoxifen is preferred for {{c1::premenopausal}} women, AIs for {{c1::postmenopausal}} women"
→ Both use c1 since they test the same concept (menopause status for drug choice)

❌ POOR: "Adjuvant therapy is recommended for {{c1::all}} ER-positive cancers"
→ Only tests "all" vs "some" - too simple
✅ BETTER: "Adjuvant {{c1::endocrine}} therapy for {{c2::ER-positive}} early breast cancer is given for {{c3::5 years}} (10 years if high-risk)"
→ Tests therapy type, receptor status, and duration

❌ INCOMPLETE: "Early breast cancer is {{c1::potentially curable}}"
→ Misses key definitional fact
✅ COMPLETE: "Early breast cancer is confined to {{c1::breast ± axillary nodes}} and is {{c2::potentially curable}}"
→ Tests both definition and prognosis

❌ SUPERFICIAL: "{{c1::Palbociclib}} blocks G1-to-S transition"
→ Only tests drug name
✅ COMPREHENSIVE: "{{c1::CDK4/6 inhibitors}} like {{c2::palbociclib}} block {{c3::G1-to-S phase transition}}, arresting {{c4::proliferation}} of ER+ cells"
→ Tests drug class, example, mechanism, and effect"""
    
    @staticmethod
    def get_percentage_guidelines() -> str:
        """Get guidelines for handling percentages and statistics."""
        return """
PERCENTAGES AND STATISTICS:
❌ AVOID: "BRCA1 mutations occur in {{c1::5-10%}} of breast cancers"
→ Exact percentages are hard to remember and often change
✅ BETTER: "{{c1::BRCA1}} mutations are found in 5-10% of breast cancers"
→ Tests the gene name, not the percentage
✅ OR: "BRCA1 mutations are {{c1::uncommon}}, occurring in 5-10% of breast cancers"
→ Tests clinical significance rather than exact number
✅ EXCEPTION: "BRCA1 mutations increase lifetime breast cancer risk to {{c1::60-80%}}"
→ This percentage is clinically critical for counseling patients"""
    
    @staticmethod
    def get_content_focus() -> str:
        """Get guidelines for what content to focus on."""
        return """
Your task is to create flashcards appropriate for medical student level:

1. Extract MEDICAL FACTS AND CONCEPTS that medical students need to know for exams and clinical practice (not meta-information about the lecture)
2. Create cloze deletion flashcards focusing on:
   - Core pathophysiology and disease mechanisms
   - Key clinical features and presentations
   - First-line treatments and management principles
   - Important differential diagnoses
   - High-yield diagnostic approaches
   - Clinical decision-making concepts
   - Best practice guidelines (not minute details)
   - Important contraindications and safety considerations

3. SKIP slides that only contain:
   - Title/topic announcements
   - Learning objectives/outcomes lists
   - Lecture outlines or agendas
   - Speaker introductions
   - References/bibliography

4. AVOID creating cards for:
   - Specific radiation doses or technical parameters
   - Names/authors of individual studies (unless landmark trials)
   - Overly specialized procedural details
   - Research methodology minutiae
   - Historical facts unless clinically relevant
   - Subspecialty-specific technical details

5. ENSURE each card:
   - Can be understood without seeing the original lecture
   - Has specific, unambiguous cloze deletions
   - Tests one clear concept per cloze
   - Provides enough context to identify the answer
   - Emphasizes "why" and "when" rather than exact numbers
   - Focuses on clinical reasoning and decision pathways
   - Highlights comparative effectiveness (Drug A vs Drug B)
   - Includes practical clinical applications
   - Tests ALL key facts in a statement (definitions, mechanisms, effects)
   - Uses same cloze number (c1) when testing related binary choices
   - Avoids overly simple clozes like "all" vs "some"
   - Avoids testing exact percentages unless clinically critical (prefer testing the condition/gene/drug name or using "common/rare")"""
    
    @staticmethod
    def get_json_format() -> str:
        """Get the expected JSON format for responses."""
        return """
Format your response as a JSON array of flashcard objects, where each object has:
- "text": The complete text with cloze deletions in {{c1::answer}} format (can have multiple clozes {{c1::}}, {{c2::}}, etc.)
- "facts": Array of the key facts being tested
- "context": Brief context about why this is important FOR A MEDICAL STUDENT
- "clinical_relevance": Optional field for clinical pearls or practical applications

Example:
[
  {
    "text": "In type 2 diabetes, {{c1::Metformin}} is the first-line medication because it {{c2::does not cause hypoglycemia}} and has {{c3::cardiovascular benefits}}",
    "facts": ["Metformin", "does not cause hypoglycemia", "cardiovascular benefits"],
    "context": "Essential knowledge for diabetes management in primary care",
    "clinical_relevance": "Always check renal function before prescribing"
  }
]

Create concise, self-contained cards testing PRACTICAL MEDICAL KNOWLEDGE with clear, unambiguous cloze deletions."""


class MedicalAnkiGenerator:
    def __init__(self, openai_api_key: str, single_card_mode: bool = False, 
                 custom_tags: Optional[List[str]] = None, card_style: Optional[Dict] = None,
                 batch_mode: bool = False, compression_level: str = "none"):
        self.api_key = openai_api_key
        self.single_card_mode = single_card_mode
        self.custom_tags = custom_tags or []
        self.card_style = card_style or {}
        self.batch_mode = batch_mode
        self.compression_level = compression_level
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Setup logging
        self.setup_logging()
        
        # Compression settings
        self.compression_settings = {
            "none": {"max_size": 1024, "quality": 95, "format": "PNG"},
            "low": {"max_size": 1024, "quality": 90, "format": "JPEG"},
            "medium": {"max_size": 800, "quality": 85, "format": "JPEG"},
            "high": {"max_size": 512, "quality": 80, "format": "JPEG"}
        }
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Create a custom Anki model with enhanced styling
        self.cloze_model = self._create_styled_model()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("anki_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ankify_log_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ankify session started")
        self.logger.info(f"Configuration: single_card={self.single_card_mode}, batch={self.batch_mode}, compression={self.compression_level}")
        
    def _create_styled_model(self):
        """Create Anki model with custom styling."""
        # Default style values
        bg_color = self.card_style.get('background', 'white')
        text_color = self.card_style.get('text_color', 'black')
        cloze_color = self.card_style.get('cloze_color', 'blue')
        font_family = self.card_style.get('font_family', 'arial')
        font_size = self.card_style.get('font_size', '20px')
        
        # Create a unique model ID based on style settings
        style_hash = hash(str(self.card_style))
        model_id = 1234567890 + (abs(style_hash) % 1000000)
        
        # Create model name that reflects custom styling
        style_desc = []
        if bg_color != 'white':
            style_desc.append('Custom BG')
        if cloze_color != 'blue':
            style_desc.append('Custom Cloze')
        model_name = 'Medical Cloze with Image' + (' (' + ', '.join(style_desc) + ')' if style_desc else '')
        
        css = f'''
            .card {{
                font-family: {font_family};
                font-size: {font_size};
                text-align: center;
                color: {text_color};
                background-color: {bg_color};
                padding: 20px;
            }}
            .cloze {{
                font-weight: bold;
                color: {cloze_color};
                background-color: rgba(255, 255, 255, 0.1);
                padding: 2px 4px;
                border-radius: 3px;
            }}
            img {{
                max-width: 100%;
                max-height: 600px;
                margin-top: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .context {{
                font-style: italic;
                color: {self._adjust_color_brightness(text_color, 0.7)};
                margin-top: 15px;
                font-size: 0.9em;
            }}
            b, strong {{
                color: {self._adjust_color_brightness(cloze_color, 1.2)};
                font-weight: bold;
            }}
            .clinical-pearl {{
                background-color: rgba(255, 243, 205, 0.3);
                border: 1px solid rgba(255, 234, 167, 0.5);
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
                text-align: left;
                color: {text_color};
            }}
            /* Night mode support */
            .night_mode .card {{
                color: {text_color};
                background-color: {bg_color};
            }}
            .night_mode .cloze {{
                color: {cloze_color};
            }}
        '''
        
        return genanki.Model(
            model_id,
            model_name,
            fields=[
                {'name': 'Text'},
                {'name': 'Extra'},
            ],
            templates=[
                {
                    'name': 'Cloze',
                    'qfmt': '{{cloze:Text}}',
                    'afmt': '{{cloze:Text}}<br><br>{{Extra}}',
                },
            ],
            css=css,
            model_type=genanki.Model.CLOZE
        )
    
    def _adjust_color_brightness(self, color: str, factor: float) -> str:
        """Adjust color brightness for better contrast."""
        if not color.startswith('#'):
            return color
        
        try:
            color = color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return color
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[Tuple[Image.Image, int]]:
        """Convert PDF pages to images."""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append((img, page_num + 1))
        
        doc.close()
        return images
    
    def image_to_base64(self, image: Image.Image, apply_compression: bool = True) -> str:
        """Convert PIL Image to base64 string with optional compression."""
        buffered = io.BytesIO()
        
        if apply_compression and self.compression_level != "none":
            settings = self.compression_settings[self.compression_level]
            max_size = settings["max_size"]
            quality = settings["quality"]
            img_format = settings["format"]
            
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            if img_format == "JPEG" and image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            
            try:
                image.save(buffered, format=img_format, quality=quality, optimize=True)
            except Exception as e:
                print(f"⚠️ Compression failed, using original: {str(e)}")
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
        else:
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            image.save(buffered, format="PNG", optimize=True)
        
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def escape_html_but_preserve_formatting(self, text: str) -> str:
        """Escape HTML characters but preserve our formatting tags."""
        text = html.escape(text, quote=False)
        
        # Restore specific formatting tags
        replacements = [
            ('&lt;b&gt;', '<b>'), ('&lt;/b&gt;', '</b>'),
            ('&lt;strong&gt;', '<strong>'), ('&lt;/strong&gt;', '</strong>'),
            ('&lt;i&gt;', '<i>'), ('&lt;/i&gt;', '</i>'),
            ('&lt;em&gt;', '<em>'), ('&lt;/em&gt;', '</em>')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def convert_to_single_card_format(self, text: str) -> str:
        """Convert multiple cloze numbers (c1, c2, c3...) to all c1 for single card mode."""
        if self.single_card_mode:
            return re.sub(r'\{\{c\d+::', '{{c1::', text)
        return text
    
    def add_bold_formatting(self, text: str) -> str:
        """Add bold formatting to key medical terms not in cloze deletions."""
        key_patterns = [
            r'\b(diagnosis|treatment|syndrome|disease|disorder|symptom|sign|pathophysiology|mechanism|receptor|enzyme|hormone|drug|medication|dose|contraindication|indication|complication|prognosis|etiology|differential|investigation|management)\b',
            r'\b(acute|chronic|primary|secondary|benign|malignant|systemic|focal|diffuse|bilateral|unilateral)\b',
            r'\b(\d+\s*(?:mg|mcg|g|kg|mL|L|mmHg|bpm|/min|/hr|/day|%|mmol|mg/dL))\b'
        ]
        
        def replace_if_not_in_cloze(match):
            term = match.group(0)
            start = match.start()
            before_text = text[:start]
            if before_text.count('{{') > before_text.count('}}'):
                return term
            return f'<b>{term}</b>'
        
        for pattern in key_patterns:
            text = re.sub(pattern, replace_if_not_in_cloze, text, flags=re.IGNORECASE)
        
        return text
    
    def _build_analysis_prompt(self, page_num: int, lecture_name: str) -> str:
        """Build the analysis prompt using centralized templates."""
        cloze_instruction = PromptTemplates.get_cloze_instruction(self.single_card_mode)
        
        return f"""You are analyzing slide {page_num} from a medical lecture on "{lecture_name}" for MEDICAL STUDENTS.
        
{PromptTemplates.get_common_rules()}

{PromptTemplates.get_cloze_examples()}

{PromptTemplates.get_advanced_principles()}

{PromptTemplates.get_percentage_guidelines()}

{PromptTemplates.get_content_focus()}

{cloze_instruction}

{PromptTemplates.get_json_format()}"""
    
    def _build_batch_analysis_prompt(self, num_slides: int, lecture_name: str) -> str:
        """Build the batch analysis prompt using centralized templates."""
        cloze_instruction = PromptTemplates.get_cloze_instruction(self.single_card_mode)
        
        return f"""You are analyzing {num_slides} slides from a medical lecture on "{lecture_name}" for MEDICAL STUDENTS.
        
{PromptTemplates.get_common_rules()}

{PromptTemplates.get_cloze_examples()}

{PromptTemplates.get_advanced_principles()}

{PromptTemplates.get_percentage_guidelines()}

For EACH slide, extract MEDICAL FACTS AND CONCEPTS that medical students need to know for exams and clinical practice (not meta-information about the lecture).

SKIP slides that only contain:
- Title/topic announcements
- Learning objectives/outcomes lists
- Lecture outlines or agendas
- Speaker introductions
- References/bibliography

FOCUS on creating cards for:
- Core pathophysiology and disease mechanisms
- Key clinical features and presentations
- First-line treatments and management principles
- Important differential diagnoses
- High-yield diagnostic approaches
- Clinical decision-making concepts
- Best practice guidelines (not minute details)
- Important contraindications and safety

AVOID cards for:
- Specific radiation doses or technical parameters
- Names/authors of individual studies (unless landmark trials)
- Overly specialized procedural details
- Research methodology minutiae
- Subspecialty-specific technical details

Emphasize:
- "Why" and "when" rather than exact numbers
- Clinical reasoning and decision pathways
- Comparative effectiveness
- Practical clinical applications

ENSURE each card:
- Can be understood without seeing the original lecture
- Has specific, unambiguous cloze deletions
- Tests one clear concept per cloze
- Provides enough context to identify the answer
- Tests ALL key facts (definitions, mechanisms, effects, durations)
- Uses same cloze number (c1) for related binary/mutually exclusive choices
- Avoids overly simple clozes like "all" vs "some"
- Avoids testing exact percentages unless clinically critical (prefer testing the subject or using "common/rare")

{cloze_instruction}

Return a JSON array with one object per slide:
[
  {{
    "page_num": 1,
    "cards": [
      {{
        "text": "In type 2 diabetes, {{{{c1::Metformin}}}} is first-line because it {{{{c2::doesn't cause hypoglycemia}}}}",
        "facts": ["Metformin", "doesn't cause hypoglycemia"],
        "context": "Essential diabetes management knowledge",
        "clinical_relevance": "Check renal function before prescribing"
      }}
    ]
  }}
]

IMPORTANT: Include ALL slides in your response, even if a slide has no relevant medical content (return empty cards array for that slide).
Make cards self-contained with clear, unambiguous cloze deletions that can be answered without lecture context."""
    
    def analyze_slide_with_ai(self, image: Image.Image, page_num: int, lecture_name: str, max_retries: int = 5) -> Dict:
        """Send slide image to OpenAI API for analysis with retry logic."""
        base64_image = self.image_to_base64(image)
        prompt = self._build_analysis_prompt(page_num, lecture_name)
        
        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_completion_tokens": 100000
        }
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"\n  ⏳ Retry {attempt}/{max_retries} after {wait_time:.1f}s wait...", end='', flush=True)
                    time.sleep(wait_time)
                
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        cards_data = json.loads(json_match.group())
                        if self.single_card_mode:
                            for card in cards_data:
                                card['text'] = self.convert_to_single_card_format(card['text'])
                        for card in cards_data:
                            card['text'] = self.add_bold_formatting(card['text'])
                        return {"page_num": page_num, "cards": cards_data}
                elif response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"\n  ⚠️ Rate limited. Waiting {wait_time}s...", end='', flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"\n  ❌ API Error: {response.status_code} - {response.text[:100]}...", end='', flush=True)
                    
            except Exception as e:
                print(f"\n  ❗ Error (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}...", end='', flush=True)
        
        print(f"\n  ❌ Failed after {max_retries} attempts", end='', flush=True)
        return {"page_num": page_num, "cards": []}
    
    def analyze_slides_batch(self, images: List[Tuple[Image.Image, int]], lecture_name: str, max_retries: int = 3) -> List[Dict]:
        """Send multiple slides to OpenAI API in a single batch request."""
        print(f"\n🔄 Batch processing {len(images)} slides in a single API call...")
        self.logger.info(f"Starting batch processing of {len(images)} slides")
        
        # Prepare all images
        slides_data = []
        for img, page_num in images:
            base64_image = self.image_to_base64(img, apply_compression=True)
            slides_data.append({"page_num": page_num, "base64": base64_image})
            
        self.logger.info(f"Prepared {len(slides_data)} images for batch processing")
        
        # Build prompt with all slides
        slides_content = []
        for slide in slides_data:
            slides_content.extend([
                {"type": "text", "text": f"SLIDE {slide['page_num']}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{slide['base64']}"}}
            ])
        
        prompt = self._build_batch_analysis_prompt(len(images), lecture_name)
        content = [{"type": "text", "text": prompt}] + slides_content
        
        payload = {
            "model": "o3",
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 100000
        }
        
        self.logger.info(f"Sending batch request with {len(content)} content items")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"\n  ⏳ Retry {attempt}/{max_retries} after {wait_time:.1f}s wait...", end='', flush=True)
                    time.sleep(wait_time)
                
                print(f"\n  📡 Sending API request (attempt {attempt + 1}/{max_retries})...", end='', flush=True)
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=300
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    content = response_json['choices'][0]['message']['content']
                    self.logger.info(f"Received response of length: {len(content)}")
                    
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        try:
                            all_slides_data = json.loads(json_match.group())
                            self.logger.info(f"Successfully parsed JSON with {len(all_slides_data)} slide entries")
                            
                            # Ensure we have data for all slides
                            slide_nums_in_response = {item.get('page_num', 0) for item in all_slides_data}
                            expected_slide_nums = set(range(1, len(images) + 1))
                            missing_slides = expected_slide_nums - slide_nums_in_response
                            
                            if missing_slides:
                                self.logger.warning(f"Missing slides in response: {missing_slides}")
                                for slide_num in missing_slides:
                                    all_slides_data.append({"page_num": slide_num, "cards": []})
                            
                            all_slides_data.sort(key=lambda x: x.get('page_num', 0))
                            
                            # Process and format the results
                            processed_results = []
                            total_cards = 0
                            for slide_data in all_slides_data:
                                cards = slide_data.get('cards', [])
                                if self.single_card_mode:
                                    for card in cards:
                                        card['text'] = self.convert_to_single_card_format(card['text'])
                                for card in cards:
                                    card['text'] = self.add_bold_formatting(card['text'])
                                
                                processed_results.append({
                                    "page_num": slide_data.get('page_num', 1),
                                    "cards": cards
                                })
                                total_cards += len(cards)
                            
                            print(f"\n✅ Batch processing complete: {total_cards} cards generated from {len(images)} slides")
                            self.logger.info(f"Batch processing successful: {total_cards} cards from {len(images)} slides")
                            return processed_results
                            
                        except json.JSONDecodeError as e:
                            print(f"\n  ❌ JSON parsing error: {str(e)[:100]}...")
                            self.logger.error(f"JSON parsing failed: {str(e)}")
                            self.logger.debug(f"Raw content: {content[:500]}...")
                    else:
                        print(f"\n  ❌ No JSON array found in response")
                        self.logger.error("No JSON array found in API response")
                        self.logger.debug(f"Response content: {content[:500]}...")
                else:
                    print(f"\n  ❌ API Error: {response.status_code}")
                    self.logger.error(f"Batch API error: {response.status_code} - {response.text[:500]}")
                    
            except requests.exceptions.Timeout:
                print(f"\n  ⏱️ Request timeout")
                self.logger.error("Batch request timeout")
            except requests.exceptions.ConnectionError as e:
                print(f"\n  🔌 Connection error: {str(e)[:100]}...")
                self.logger.error(f"Connection error: {str(e)}")
            except Exception as e:
                print(f"\n  ❗ Unexpected error: {str(e)[:100]}...")
                self.logger.error(f"Unexpected batch processing error: {str(e)}", exc_info=True)
        
        print(f"\n❌ Batch processing failed after {max_retries} attempts")
        self.logger.error("Batch processing failed completely")
        return []
    
    def critique_and_refine_cards(self, all_cards_data: List[Dict], lecture_name: str) -> List[Dict]:
        """Use AI model to critique and refine all cards for optimal learning."""
        print("\n🔬 Starting advanced critique and refinement pass...")
        self.logger.info(f"Starting critique for lecture: {lecture_name}")
        
        # Create refinement log file
        refinement_log_dir = Path("anki_logs") / "refinements"
        refinement_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        refinement_log_file = refinement_log_dir / f"{lecture_name}_refinement_{timestamp}.json"
        
        # Prepare all cards for critique
        cards_for_review = []
        total_original_cards = 0
        for slide_data in all_cards_data:
            for card in slide_data['cards']:
                cards_for_review.append({
                    'slide': slide_data['page_num'],
                    'text': card['text'],
                    'facts': card.get('facts', []),
                    'context': card.get('context', ''),
                    'clinical_relevance': card.get('clinical_relevance', ''),
                    'original_index': total_original_cards
                })
                total_original_cards += 1
        
        print(f"📊 Analyzing {total_original_cards} cards for optimization...")
        self.logger.info(f"Total cards to analyze: {total_original_cards}")
        
        prompt = self._build_critique_prompt(lecture_name, cards_for_review)
        
        payload = {
            "model": "o3",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 100000
        }
        
        try:
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = json.loads(json_match.group())
                    refined_cards = result.get('refined_cards', [])
                    decisions = result.get('decisions', [])
                    
                    # Save and process refinement results
                    self._save_refinement_logs(refinement_log_file, lecture_name, total_original_cards, 
                                             refined_cards, decisions)
                    
                    # Validate and organize refined cards
                    return self._process_refined_cards(refined_cards)
                    
        except Exception as e:
            self.logger.error(f"Refinement failed: {str(e)}", exc_info=True)
            print(f"❌ Refinement failed: {str(e)}")
            print("⚠️ Using original cards without refinement")
            import traceback
            traceback.print_exc()
        
        return all_cards_data
    
    def _build_critique_prompt(self, lecture_name: str, cards_for_review: List[Dict]) -> str:
        """Build the critique prompt using centralized templates."""
        cloze_format_instruction = "using {{c1::}}, {{c2::}}, etc." if not self.single_card_mode else "using ONLY {{c1::}} for all clozes"
        
        return f"""You are an expert medical educator reviewing cloze deletion flashcards from a lecture on "{lecture_name}".

CRITICAL INSTRUCTIONS:
1. ALL cards MUST remain in cloze deletion format {cloze_format_instruction}
2. PRESERVE the cloze deletion syntax exactly - do not convert to Q&A format
3. Each refined card must have at least one cloze deletion
4. Cards will be reviewed WITHOUT lecture context - ensure self-contained clarity
5. Keep content at MEDICAL STUDENT level - practical knowledge over specialist minutiae

⚠️ CRITICAL: MAINTAIN CLOZE DELETION QUALITY ⚠️
The original cards follow good cloze deletion patterns. When refining, you MUST preserve these qualities:

{PromptTemplates.get_cloze_examples()}

GOOD PATTERNS TO MAINTAIN (Additional Examples):
✅ "Metastatic ER+ cancer recurs {{{{c1::5-15 years}}}} post-diagnosis and metastasizes to {{{{c2::bone}}}}"
   - Two disparate facts as separate clozes

BAD PATTERNS TO AVOID CREATING (Additional Examples):
❌ "Tamoxifen for {{{{c1::premenopausal}}}}, AIs for {{{{c2::postmenopausal}}}}"
   - c2 reveals c1 (binary choice) - use same cloze number instead
❌ "Therapy is recommended for {{{{c1::all}}}} ER+ cancers"
   - Too simple, only tests "all" vs "some"

REFINEMENT RULES:
1. If a card already has good cloze patterns, DO NOT make clozes more ambiguous
2. If a card lacks context, ADD context rather than hiding more information
3. If a cloze is too broad, SPLIT it into smaller, specific clozes
4. NEVER combine multiple small clozes into one large ambiguous cloze
5. PRESERVE the self-contained nature of cards - they must work without lecture context
6. For binary/mutually exclusive choices (pre/post-menopausal), use SAME cloze number
7. Ensure ALL key facts are tested (definitions, mechanisms, effects, durations)
8. Avoid trivial clozes - test meaningful medical knowledge
9. Avoid testing exact percentages unless clinically critical - prefer testing the subject (gene/condition name) or using descriptors like "common/rare/most common"

Review these {len(cards_for_review)} cloze deletion flashcards and optimize them by:

1. MAINTAINING good cloze patterns - don't make them more ambiguous
2. ENSURING each card remains self-contained and understandable
3. FIXING only genuinely ambiguous cloze deletions
4. ADDING context where needed rather than hiding information
5. MERGING only truly redundant cards
6. REMOVING only:
   - Cards about titles, objectives, or outlines themselves
   - Cards with unfixable ambiguity
   - Overly specialized technical details
   - Specific research study names/authors (unless landmark)
   - Exact dosing/technical parameters (unless critical safety info)
   - Research methodology minutiae
   - Historical trivia without clinical relevance
   - Subspecialty procedural minutiae
7. PRESERVING:
   - Specific, focused cloze deletions
   - Clear context around clozes
   - Self-contained card structure
   - Learning outcomes focus
8. EMPHASIZING:
   - Clinical reasoning and decision-making
   - Comparative effectiveness (why choose A over B)
   - Practical applications in general practice
   - Key safety considerations
   - First-line approaches
   - "Why" and "when" rather than exact numbers
   - Clinical decision pathways
9. ADDING clinical pearls that help with real patient care
10. ENSURING medical accuracy while keeping appropriate depth

Current flashcards:
{json.dumps(cards_for_review, indent=2)}

Return a JSON object with TWO arrays:
{{
  "refined_cards": [
    {{
      "slide": 1,
      "text": "In hereditary breast cancer, {{{{c1::BRCA1}}}} mutations increase lifetime risk to {{{{c2::60-80%}}}}",
      "facts": ["BRCA1", "60-80%"],
      "context": "Key genetic risk factor for breast cancer screening decisions",
      "clinical_relevance": "Indicates need for enhanced surveillance or prophylactic measures",
      "original_indices": [0]
    }}
  ],
  "decisions": [
    {{
      "action": "removed",
      "original_index": 1,
      "original_text": "This lecture focuses on the {{{{c1::treatment of hormone-receptor-positive breast cancer}}}}",
      "reason": "Tests the lecture title/topic announcement rather than medical facts"
    }},
    {{
      "action": "modified", 
      "original_index": 3,
      "original_text": "Treatment includes {{{{c1::chemotherapy}}}}",
      "new_text": "First-line treatment for HER2+ breast cancer includes {{{{c1::trastuzumab}}}} with {{{{c2::chemotherapy}}}}",
      "reason": "Added specific context (HER2+) to make card self-contained, split into two clozes"
    }}
  ]
}}

⚠️ REMEMBER: The goal is to REFINE cards while MAINTAINING their good cloze deletion patterns. Do not make cards more ambiguous in the name of brevity. Each card must be answerable without having seen the lecture."""
    
    def _save_refinement_logs(self, refinement_log_file: Path, lecture_name: str, 
                            total_original_cards: int, refined_cards: List[Dict], 
                            decisions: List[Dict]):
        """Save refinement logs in both JSON and human-readable formats."""
        refinement_data = {
            "lecture": lecture_name,
            "timestamp": datetime.now().isoformat(),
            "original_count": total_original_cards,
            "refined_count": len(refined_cards),
            "decisions": decisions,
            "summary": {
                "removed": len([d for d in decisions if d.get('action') == 'removed']),
                "merged": len([d for d in decisions if d.get('action') == 'merged']),
                "modified": len([d for d in decisions if d.get('action') == 'modified'])
            }
        }
        
        # Save JSON log
        with open(refinement_log_file, 'w', encoding='utf-8') as f:
            json.dump(refinement_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Refinement decisions logged to: {refinement_log_file}")
        self.logger.info(f"Refinement complete: {total_original_cards} → {len(refined_cards)} cards")
        self.logger.info(f"Removed: {refinement_data['summary']['removed']}, "
                        f"Merged: {refinement_data['summary']['merged']}, "
                        f"Modified: {refinement_data['summary']['modified']}")
        
        # Create human-readable summary
        summary_file = refinement_log_file.with_suffix('.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Refinement Summary for {lecture_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original cards: {total_original_cards}\n")
            f.write(f"Refined cards: {len(refined_cards)}\n")
            if total_original_cards > 0:
                reduction_percent = (1 - len(refined_cards)/total_original_cards) * 100
                f.write(f"Reduction: {total_original_cards - len(refined_cards)} cards ({reduction_percent:.1f}%)\n\n")
            else:
                f.write(f"Reduction: 0 cards (0.0%)\n\n")
            
            f.write("DECISIONS:\n")
            f.write("-"*60 + "\n\n")
            
            for decision in decisions:
                f.write(f"ACTION: {decision['action'].upper()}\n")
                if decision['action'] == 'removed':
                    f.write(f"Card #{decision['original_index']}: {decision.get('original_text', 'N/A')[:100]}...\n")
                elif decision['action'] == 'merged':
                    f.write(f"Cards #{decision['original_indices']}\n")
                elif decision['action'] == 'modified':
                    f.write(f"Card #{decision['original_index']}\n")
                    f.write(f"Original: {decision.get('original_text', 'N/A')[:100]}...\n")
                    f.write(f"New: {decision.get('new_text', 'N/A')[:100]}...\n")
                f.write(f"REASON: {decision['reason']}\n")
                f.write("-"*40 + "\n\n")
        
        print(f"📄 Human-readable summary saved to: {summary_file}")
    
    def _process_refined_cards(self, refined_cards: List[Dict]) -> List[Dict]:
        """Process and organize refined cards back into slide structure."""
        # Validate that cards still have cloze format
        valid_cards = []
        for card in refined_cards:
            if '{{c' in card.get('text', ''):
                valid_cards.append(card)
            else:
                self.logger.warning(f"Skipping card without cloze format: {card.get('text', '')[:50]}...")
        
        # Reorganize refined cards back into slide structure
        refined_data = {}
        for card in valid_cards:
            slide_num = card.get('slide', 1)
            if slide_num not in refined_data:
                refined_data[slide_num] = {
                    'page_num': slide_num,
                    'cards': []
                }
            
            # Apply single card format if needed
            card_text = card['text']
            if self.single_card_mode:
                card_text = self.convert_to_single_card_format(card_text)
            
            refined_data[slide_num]['cards'].append({
                'text': self.add_bold_formatting(card_text),
                'facts': card.get('facts', []),
                'context': card.get('context', ''),
                'clinical_relevance': card.get('clinical_relevance', '')
            })
        
        refined_list = list(refined_data.values())
        total_refined_cards = sum(len(d['cards']) for d in refined_list)
        
        print(f"✅ Refinement complete: {len(refined_cards)} cards → {total_refined_cards} optimized cards")
        
        return refined_list
    
    def save_progress(self, progress_file: Path, progress_data: Dict):
        """Save progress to a file."""
        with open(progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
    
    def load_progress(self, progress_file: Path) -> Optional[Dict]:
        """Load progress from a file."""
        if progress_file.exists():
            try:
                with open(progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Could not load progress file: {e}")
        return None
    
    def create_anki_package(self, cards_data: List[Dict], lecture_name: str, images: List[Tuple[Image.Image, int]], 
                          output_dir: str, deck_suffix: str = ""):
        """Create Anki package (.apkg) with cards and images using genanki."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        deck_id = random.randrange(1 << 30, 1 << 31)
        deck_name = f'Medical::{lecture_name}{deck_suffix}'
        deck = genanki.Deck(deck_id, deck_name)
        
        media_files = []
        page_to_image = {page_num: img for img, page_num in images}
        
        all_cards_text = []
        card_number = 1
        
        temp_media_dir = output_path / "temp_media"
        temp_media_dir.mkdir(exist_ok=True)
        
        card_mode_text = "Single Card Mode (all blanks shown together)" if self.single_card_mode else "Multiple Card Mode (separate cards for each blank)"
        all_cards_text.append(f"Card Mode: {card_mode_text}")
        if deck_suffix:
            all_cards_text.append(f"Deck Type: {deck_suffix.replace('::', '').strip()}")
        if self.custom_tags:
            all_cards_text.append(f"Custom Tags: {', '.join(self.custom_tags)}")
        all_cards_text.append("-" * 50)
        
        for slide_data in cards_data:
            page_num = slide_data['page_num']
            slide_cards = slide_data['cards']
            
            image_filename = f"slide_{lecture_name}_{page_num:03d}.png"
            if page_num in page_to_image:
                image_path = temp_media_dir / image_filename
                if not image_path.exists():
                    page_to_image[page_num].save(image_path, "PNG", optimize=True)
                media_files.append(str(image_path))
            
            for card in slide_cards:
                note_text = card['text']
                note_text = self.escape_html_but_preserve_formatting(note_text)
                
                # Build extra content
                extra_parts = [f'<img src="{image_filename}">']
                if card.get('clinical_relevance'):
                    clinical_text = html.escape(card['clinical_relevance'])
                    extra_parts.append(f'<div class="clinical-pearl">💡 {clinical_text}</div>')
                context_text = html.escape(card.get('context', ''))
                extra_parts.append(f'<div class="context">Context: {context_text}</div>')
                extra_content = '<br>'.join(extra_parts)
                
                # Combine default and custom tags
                tags = [f'slide_{page_num}', lecture_name.replace(" ", "_"), 'medical'] + self.custom_tags
                if deck_suffix:
                    tags.append(deck_suffix.replace('::', '').strip().lower())
                
                note = genanki.Note(
                    model=self.cloze_model,
                    fields=[note_text, extra_content],
                    tags=tags
                )
                deck.add_note(note)
                
                # Add to text file
                all_cards_text.append(f"Card {card_number} (Slide {page_num}):")
                all_cards_text.append(f"Text: {note_text}")
                all_cards_text.append(f"Facts tested: {', '.join(card.get('facts', []))}")
                all_cards_text.append(f"Context: {card.get('context', 'N/A')}")
                if card.get('clinical_relevance'):
                    all_cards_text.append(f"Clinical Relevance: {card['clinical_relevance']}")
                all_cards_text.append(f"Tags: {', '.join(tags)}")
                all_cards_text.append("-" * 50)
                
                card_number += 1
        
        package = genanki.Package(deck)
        package.media_files = media_files
        
        # Create filename with suffix
        filename_suffix = deck_suffix.replace('::', '_').strip() if deck_suffix else ""
        apkg_filename = output_path / f"{lecture_name}{filename_suffix}.apkg"
        package.write_to_file(str(apkg_filename))
        
        text_file = output_path / f"{lecture_name}{filename_suffix}_cards_reference.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Anki Cards for {lecture_name}{deck_suffix}\n")
            f.write(f"Total cards: {card_number - 1}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(all_cards_text))
        
        print(f"\n✅ Successfully created {card_number - 1} flashcards")
        if self.card_style:
            print(f"🎨 Custom styling applied: {', '.join([f'{k}={v}' for k, v in self.card_style.items()])}")
        print(f"📦 Anki package saved: {apkg_filename}")
        print(f"📄 Reference text saved: {text_file}")
        
        return apkg_filename
    
    def process_lecture(self, pdf_path: str, output_dir: str = "anki_output", resume: bool = True, 
                       advanced_mode: bool = False):
        """Process a single lecture PDF with resume capability."""
        lecture_name = Path(pdf_path).stem
        print(f"\n🔍 Processing lecture: {lecture_name}")
        self.logger.info(f"Processing lecture: {lecture_name} from {pdf_path}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        if self.batch_mode:
            print(f"📦 Batch mode: Enabled (compression: {self.compression_level})")
        if advanced_mode:
            print("🧠 Advanced mode: Enabled (will critique and refine cards)")
        
        progress_dir = Path(output_dir) / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_file = progress_dir / f"{lecture_name}_progress.pkl"
        
        progress_data = None
        if resume:
            progress_data = self.load_progress(progress_file)
            if progress_data:
                print(f"📂 Found existing progress: {len(progress_data['completed_slides'])} slides already processed")
        
        print("📄 Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)
        print(f"✅ Extracted {len(images)} slides")
        
        if progress_data is None:
            progress_data = {
                'lecture_name': lecture_name,
                'total_slides': len(images),
                'completed_slides': [],
                'cards_data': [],
                'start_time': datetime.now().isoformat(),
                'single_card_mode': self.single_card_mode
            }
        
        all_cards_data = progress_data['cards_data']
        completed_slides = set(progress_data['completed_slides'])
        
        # Process slides based on mode
        if self.batch_mode:
            remaining_images = [(img, page_num) for img, page_num in images if page_num not in completed_slides]
            
            if remaining_images:
                print(f"\n🔄 Batch processing {len(remaining_images)} remaining slides...")
                batch_results = self.analyze_slides_batch(remaining_images, lecture_name)
                
                if batch_results:
                    for slide_data in batch_results:
                        if slide_data['cards']:
                            all_cards_data.append(slide_data)
                        completed_slides.add(slide_data['page_num'])
                    
                    progress_data['completed_slides'] = list(completed_slides)
                    progress_data['cards_data'] = all_cards_data
                    progress_data['last_update'] = datetime.now().isoformat()
                    self.save_progress(progress_file, progress_data)
                else:
                    print("⚠️ Batch processing failed, falling back to individual processing...")
                    self._process_slides_individually(remaining_images, lecture_name, completed_slides, 
                                                    all_cards_data, progress_data, progress_file)
        else:
            remaining_images = [(img, page_num) for img, page_num in images if page_num not in completed_slides]
            self._process_slides_individually(remaining_images, lecture_name, completed_slides, 
                                            all_cards_data, progress_data, progress_file)
        
        # Handle advanced mode or normal completion
        if advanced_mode and all_cards_data:
            refined_cards_data = self.critique_and_refine_cards(all_cards_data, lecture_name)
            
            print("\n📦 Creating ORIGINAL deck...")
            original_apkg = self.create_anki_package(all_cards_data, lecture_name, images, output_dir, 
                                                   deck_suffix="::Original")
            
            print("\n📦 Creating REFINED deck...")
            refined_apkg = self.create_anki_package(refined_cards_data, lecture_name, images, output_dir, 
                                                  deck_suffix="::Refined")
            
            self._cleanup_temp_files(output_dir, progress_file)
            
            print("\n🎯 Advanced mode complete!")
            print(f"📊 Original: {sum(len(d['cards']) for d in all_cards_data)} cards")
            print(f"📊 Refined: {sum(len(d['cards']) for d in refined_cards_data)} cards")
            print("💡 Import both decks to compare and choose the best version!")
            
            return [original_apkg, refined_apkg]
        else:
            print("\n📦 Creating Anki package...")
            apkg_path = self.create_anki_package(all_cards_data, lecture_name, images, output_dir)
            
            self._cleanup_temp_files(output_dir, progress_file)
            
            return apkg_path
    
    def _process_slides_individually(self, remaining_images: List[Tuple[Image.Image, int]], 
                                   lecture_name: str, completed_slides: set, all_cards_data: List[Dict],
                                   progress_data: Dict, progress_file: Path):
        """Process slides individually with progress tracking."""
        total_slides = len(remaining_images) + len(completed_slides)
        
        for img, page_num in remaining_images:
            print(f"🤖 Analyzing slide {page_num}/{total_slides}...", end='', flush=True)
            slide_data = self.analyze_slide_with_ai(img, page_num, lecture_name)
            
            if slide_data['cards']:
                all_cards_data.append(slide_data)
                print(f" → {len(slide_data['cards'])} cards generated")
            else:
                print(" → No cards generated")
            
            completed_slides.add(page_num)
            progress_data['completed_slides'] = list(completed_slides)
            progress_data['cards_data'] = all_cards_data
            progress_data['last_update'] = datetime.now().isoformat()
            
            self.save_progress(progress_file, progress_data)
    
    def _cleanup_temp_files(self, output_dir: str, progress_file: Path):
        """Clean up temporary files after processing."""
        temp_media_dir = Path(output_dir) / "temp_media"
        if temp_media_dir.exists():
            for file in temp_media_dir.glob("*"):
                file.unlink()
            temp_media_dir.rmdir()
        
        if progress_file.exists():
            progress_file.unlink()
            print("🧹 Cleaned up progress file")
    
    def process_folder(self, folder_path: str, output_dir: str = "anki_output", resume: bool = True, 
                      advanced_mode: bool = False):
        """Process all PDFs in a folder with resume capability."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        print(f"\n📁 Found {len(pdf_files)} PDF files in {folder}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        if advanced_mode:
            print("🧠 Advanced mode: Enabled (will critique and refine cards)")
        
        progress_dir = Path(output_dir) / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        folder_progress_file = progress_dir / "folder_progress.json"
        
        completed_files = set()
        if resume and folder_progress_file.exists():
            try:
                with open(folder_progress_file, 'r') as f:
                    folder_progress = json.load(f)
                    completed_files = set(folder_progress.get('completed_files', []))
                    print(f"📂 Found folder progress: {len(completed_files)} files already completed")
            except Exception as e:
                print(f"⚠️ Could not load folder progress: {e}")
        
        successful = len(completed_files)
        for i, pdf_file in enumerate(pdf_files, 1):
            if str(pdf_file) in completed_files:
                print(f"\n⏭️ Skipping {pdf_file.name} (already completed)")
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            print(f"{'='*60}")
            
            try:
                self.process_lecture(str(pdf_file), output_dir, resume=resume, advanced_mode=advanced_mode)
                successful += 1
                
                completed_files.add(str(pdf_file))
                folder_progress = {
                    'completed_files': list(completed_files),
                    'total_files': len(pdf_files),
                    'last_update': datetime.now().isoformat(),
                    'single_card_mode': self.single_card_mode
                }
                progress_dir.mkdir(exist_ok=True)
                with open(folder_progress_file, 'w') as f:
                    json.dump(folder_progress, f, indent=2)
                    
            except Exception as e:
                print(f"\n❌ Error processing {pdf_file.name}: {str(e)}")
                print("💾 Progress saved - you can resume later")
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully processed {successful}/{len(pdf_files)} lectures")
        print(f"📁 All output saved to: {Path(output_dir).absolute()}")
        
        if successful == len(pdf_files) and folder_progress_file.exists():
            folder_progress_file.unlink()
            print("🧹 Cleaned up folder progress file")


def parse_style_options(style_string: str) -> Dict:
    """Parse style options from command line string."""
    style = {}
    if style_string:
        for option in style_string.split(','):
            if '=' in option:
                key, value = option.split('=', 1)
                style[key.strip()] = value.strip()
    return style


def parse_tags(tags_string: str) -> List[str]:
    """Parse custom tags from command line string."""
    if tags_string:
        return [tag.strip() for tag in tags_string.split(',')]
    return []


def main():
    print("""
    🏥 Ankify: Artificially Intelligent Flashcard Creation (Advanced Edition)
    ======================================================
    
    Features:
    - Automatic retry on API errors
    - Resume from interruptions
    - Single/Multiple card modes
    - Batch processing with compression
    - Custom styling and tags
    - Advanced AI critique mode
    - Bold formatting for key terms
    
    Requirements:
    1. Install: pip install pymupdf pillow requests genanki
    2. Get OpenAI API key from https://platform.openai.com/api-keys
    """)
    
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <api_key> <pdf_file_or_folder> [options]")
        print("\nOptions:")
        print("  --single-card         All blanks on one card ({{c1::}} only)")
        print("  --no-resume          Start fresh (don't resume)")
        print("  --advanced           Enable critique & refinement pass")
        print("  --batch              Process all slides in one API call")
        print("  --compress=LEVEL     Image compression (none/low/medium/high)")
        print("  --tags=tag1,tag2     Add custom tags to all cards")
        print("  --style=key=value    Custom styling (see examples)")
        print("\nCompression levels:")
        print("  none   = Original quality (default)")
        print("  low    = 1024px, JPEG 90%")
        print("  medium = 800px, JPEG 85% (recommended for batch)")
        print("  high   = 512px, JPEG 80% (maximum savings)")
        print("\nExamples:")
        print("  # Standard processing")
        print("  python script.py sk-abc... lecture.pdf")
        print("\n  # Cost-efficient batch mode")
        print("  python script.py sk-abc... lecture.pdf --batch --compress=medium")
        print("\n  # Maximum efficiency")
        print("  python script.py sk-abc... /lectures/ --batch --compress=high --single-card")
        print("\n  # Full featured")
        print("  python script.py sk-abc... lecture.pdf --advanced --tags=cardiology --style=background=#f0f0f0")
        return
    
    api_key = sys.argv[1]
    path = sys.argv[2]
    
    # Parse options
    resume = "--no-resume" not in sys.argv
    single_card_mode = "--single-card" in sys.argv
    advanced_mode = "--advanced" in sys.argv
    batch_mode = "--batch" in sys.argv
    
    # Parse compression level
    compression_level = "none"
    for arg in sys.argv:
        if arg.startswith("--compress="):
            level = arg.split("=", 1)[1]
            if level in ["none", "low", "medium", "high"]:
                compression_level = level
            else:
                print(f"⚠️ Invalid compression level '{level}', using 'none'")
    
    # Parse custom tags
    custom_tags = []
    for arg in sys.argv:
        if arg.startswith("--tags="):
            custom_tags = parse_tags(arg.split("=", 1)[1])
    
    # Parse custom style
    card_style = {}
    for arg in sys.argv:
        if arg.startswith("--style="):
            card_style = parse_style_options(arg.split("=", 1)[1])
    
    # Show cost estimate for batch mode
    if batch_mode and compression_level != "none":
        print(f"\n💰 Cost optimization: Batch mode with {compression_level} compression")
        savings_map = {'low': '30%', 'medium': '50%', 'high': '70%'}
        print(f"   Estimated cost reduction: {savings_map.get(compression_level)}+")
    
    generator = MedicalAnkiGenerator(
        api_key, 
        single_card_mode=single_card_mode,
        custom_tags=custom_tags,
        card_style=card_style,
        batch_mode=batch_mode,
        compression_level=compression_level
    )
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        generator.process_lecture(path, resume=resume, advanced_mode=advanced_mode)
    elif os.path.isdir(path):
        generator.process_folder(path, resume=resume, advanced_mode=advanced_mode)
    else:
        print("❌ Please provide a valid PDF file or folder path")


if __name__ == "__main__":
    main()