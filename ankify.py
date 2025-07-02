import os
import json
import base64
import random
import time
import pickle
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
from requests.packages.urllib3.util.retry import Retry

class MedicalAnkiGenerator:
    def __init__(self, openai_api_key: str, single_card_mode: bool = False, 
                 custom_tags: Optional[List[str]] = None, card_style: Optional[Dict] = None):
        self.api_key = openai_api_key
        self.single_card_mode = single_card_mode
        self.custom_tags = custom_tags or []
        self.card_style = card_style or {}
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
        model_id = 1234567890  # Fixed ID for consistency
        self.cloze_model = self._create_styled_model()
        
    def _create_styled_model(self):
        """Create Anki model with custom styling."""
        # Default style values
        bg_color = self.card_style.get('background', 'white')
        text_color = self.card_style.get('text_color', 'black')
        cloze_color = self.card_style.get('cloze_color', 'blue')
        font_family = self.card_style.get('font_family', 'arial')
        font_size = self.card_style.get('font_size', '20px')
        
        # Create a unique model ID based on style settings
        # This ensures Anki creates a new note type with the custom CSS
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
                    'qfmt': '{{cloze:Text}}',  # Front: Only the cloze text
                    'afmt': '{{cloze:Text}}<br><br>{{Extra}}',  # Back: Cloze text + Extra content
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
            # Convert hex to RGB
            color = color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            
            # Adjust brightness
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            
            # Convert back to hex
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
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        # Resize if too large to save on API costs
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image.save(buffered, format="PNG", optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def convert_to_single_card_format(self, text: str) -> str:
        """Convert multiple cloze numbers (c1, c2, c3...) to all c1 for single card mode."""
        if self.single_card_mode:
            # Replace all {{c[0-9]+:: with {{c1::
            return re.sub(r'\{\{c\d+::', '{{c1::', text)
        return text
    
    def add_bold_formatting(self, text: str) -> str:
        """Add bold formatting to key medical terms not in cloze deletions."""
        # List of common medical term patterns to bold
        key_patterns = [
            r'\b(diagnosis|treatment|syndrome|disease|disorder|symptom|sign|pathophysiology|mechanism|receptor|enzyme|hormone|drug|medication|dose|contraindication|indication|complication|prognosis|etiology|differential|investigation|management)\b',
            r'\b(acute|chronic|primary|secondary|benign|malignant|systemic|focal|diffuse|bilateral|unilateral)\b',
            r'\b(\d+\s*(?:mg|mcg|g|kg|mL|L|mmHg|bpm|/min|/hr|/day|%|mmol|mg/dL))\b'
        ]
        
        # Don't bold text that's already in cloze deletions
        def replace_if_not_in_cloze(match):
            term = match.group(0)
            # Check if this term is within a cloze deletion
            start = match.start()
            # Simple check - could be made more sophisticated
            before_text = text[:start]
            if before_text.count('{{') > before_text.count('}}'):
                return term  # Inside cloze, don't bold
            return f'<b>{term}</b>'
        
        for pattern in key_patterns:
            text = re.sub(pattern, replace_if_not_in_cloze, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_slide_with_ai(self, image: Image.Image, page_num: int, lecture_name: str, max_retries: int = 5) -> Dict:
        """Send slide image to OpenAI API for analysis with retry logic."""
        base64_image = self.image_to_base64(image)
        
        cloze_instruction = """IMPORTANT: Create cloze deletions using {{c1::}}, {{c2::}}, {{c3::}} etc. for different blanks within the same card."""
        if self.single_card_mode:
            cloze_instruction = """IMPORTANT: Use ONLY {{c1::}} for ALL cloze deletions (this creates a single card with multiple blanks revealed simultaneously)."""
        
        prompt = f"""You are analyzing slide {page_num} from a medical lecture on "{lecture_name}".
        
        Your task is to:
        1. Extract ALL important medical facts, concepts, definitions, and relationships that could be tested in MCQ exams
        2. Create cloze deletion flashcards for each key fact
        3. Focus on testable information like:
           - Definitions and terminology
           - Numerical values (doses, percentages, durations)
           - Classifications and categories
           - Mechanisms of action
           - Clinical features and symptoms
           - Diagnostic criteria
           - Treatment protocols
           - Anatomical relationships
           - Pathophysiology concepts
        
        {cloze_instruction}
        
        Format your response as a JSON array of flashcard objects, where each object has:
        - "text": The complete text with cloze deletions in {{c1::answer}} format (can have multiple clozes {{c1::}}, {{c2::}}, etc.)
        - "facts": Array of the key facts being tested
        - "context": Brief context about why this is important
        - "clinical_relevance": Optional field for clinical pearls or practical applications
        
        Example:
        [
          {{
            "text": "{{{{c1::Peristalsis}}}} is the {{{{c2::autonomous rhythmic contraction}}}} of smooth muscle in the GI tract",
            "facts": ["Peristalsis", "autonomous rhythmic contraction"],
            "context": "Key GI physiology concept",
            "clinical_relevance": "Understanding peristalsis is crucial for diagnosing motility disorders"
          }}
        ]
        
        Create as many cards as needed to cover all testable information on this slide. Use multiple cloze deletions in a single card when testing related concepts. **Make the cards as concise as possible while retaining the key points**"""
        
        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
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
                        # Convert to single card format if needed
                        if self.single_card_mode:
                            for card in cards_data:
                                card['text'] = self.convert_to_single_card_format(card['text'])
                        # Add bold formatting to key terms
                        for card in cards_data:
                            card['text'] = self.add_bold_formatting(card['text'])
                        return {
                            "page_num": page_num,
                            "cards": cards_data
                        }
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
    
    def critique_and_refine_cards(self, all_cards_data: List[Dict], lecture_name: str) -> List[Dict]:
        """Use o3 model to critique and refine all cards for optimal learning."""
        print("\n🔬 Starting advanced critique and refinement pass...")
        
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
                    'clinical_relevance': card.get('clinical_relevance', '')
                })
                total_original_cards += 1
        
        print(f"📊 Analyzing {total_original_cards} cards for optimization...")
        
        cloze_format_instruction = "using {{c1::}}, {{c2::}}, etc." if not self.single_card_mode else "using ONLY {{c1::}} for all clozes"
        
        prompt = f"""You are an expert medical educator reviewing cloze deletion flashcards from a lecture on "{lecture_name}".

CRITICAL INSTRUCTIONS:
1. ALL cards MUST remain in cloze deletion format {cloze_format_instruction}
2. PRESERVE the cloze deletion syntax exactly - do not convert to Q&A format
3. Each refined card must have at least one cloze deletion

Review these {len(cards_for_review)} cloze deletion flashcards and optimize them by:

1. MAINTAINING cloze format while improving clarity
2. MERGING redundant cards that test the same concept
3. SPLITTING overly complex cards
4. REMOVING low-yield information
5. ADDING clinical pearls to context when relevant
6. ENSURING medical accuracy
7. PRIORITIZING high-yield exam content

Current flashcards:
{json.dumps(cards_for_review, indent=2)}

Return a refined JSON array with the SAME structure. Each card MUST have:
- "slide": slide number
- "text": The cloze deletion text WITH {{{{c1::answer}}}} format preserved
- "facts": Array of facts being tested
- "context": Brief context
- "clinical_relevance": Optional clinical pearl

Example of correct format:
{{
  "slide": 1,
  "text": "{{{{c1::Peristalsis}}}} is the {{{{c2::rhythmic contraction}}}} of smooth muscle",
  "facts": ["Peristalsis", "rhythmic contraction"],
  "context": "Key GI physiology",
  "clinical_relevance": "Absent in ileus"
}}

DO NOT return cards like "What is peristalsis?" - they MUST have cloze deletions!"""

        payload = {
            "model": "o3",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 100000
        }
        
        try:
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300  # Longer timeout for reasoning
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    refined_cards = json.loads(json_match.group())
                    
                    # Validate that cards still have cloze format
                    valid_cards = []
                    for card in refined_cards:
                        if '{{c' in card.get('text', ''):
                            valid_cards.append(card)
                        else:
                            print(f"⚠️ Skipping card without cloze format: {card.get('text', '')[:50]}...")
                    
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
                    
                    print(f"✅ Refinement complete: {total_original_cards} cards → {total_refined_cards} optimized cards")
                    
                    # Always return refined cards in advanced mode, let process_lecture handle both versions
                    return refined_list
                    
        except Exception as e:
            print(f"❌ Refinement failed: {str(e)}")
            print("⚠️ Using original cards without refinement")
        
        return all_cards_data
    
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
    
    def create_anki_package(self, cards_data: List[Dict], lecture_name: str, images: List[Tuple[Image.Image, int]], output_dir: str, deck_suffix: str = ""):
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
                if not image_path.exists():  # Only save if not already saved
                    page_to_image[page_num].save(image_path, "PNG", optimize=True)
                media_files.append(str(image_path))
            
            for card in slide_cards:
                note_text = card['text']
                
                # Build extra content with clinical relevance if available
                extra_parts = [f'<img src="{image_filename}">']
                if card.get('clinical_relevance'):
                    extra_parts.append(f'<div class="clinical-pearl">💡 {card["clinical_relevance"]}</div>')
                extra_parts.append(f'<div class="context">Context: {card.get("context", "")}</div>')
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
    
    def process_lecture(self, pdf_path: str, output_dir: str = "anki_output", resume: bool = True, advanced_mode: bool = False):
        """Process a single lecture PDF with resume capability."""
        lecture_name = Path(pdf_path).stem
        print(f"\n🔍 Processing lecture: {lecture_name}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
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
        
        for img, page_num in images:
            if page_num in completed_slides:
                print(f"⏭️ Skipping slide {page_num}/{len(images)} (already processed)")
                continue
                
            print(f"🤖 Analyzing slide {page_num}/{len(images)}...", end='', flush=True)
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
        
        # Advanced mode: critique and refine all cards
        if advanced_mode and all_cards_data:
            refined_cards_data = self.critique_and_refine_cards(all_cards_data, lecture_name)
            
            # Create both original and refined decks
            print("\n📦 Creating ORIGINAL deck...")
            original_apkg = self.create_anki_package(all_cards_data, lecture_name, images, output_dir, deck_suffix="::Original")
            
            print("\n📦 Creating REFINED deck...")
            refined_apkg = self.create_anki_package(refined_cards_data, lecture_name, images, output_dir, deck_suffix="::Refined")
            
            # Clean up temp media only after both packages are created
            temp_media_dir = Path(output_dir) / "temp_media"
            if temp_media_dir.exists():
                for file in temp_media_dir.glob("*"):
                    file.unlink()
                temp_media_dir.rmdir()
            
            print("\n🎯 Advanced mode complete!")
            print(f"📊 Original: {sum(len(d['cards']) for d in all_cards_data)} cards")
            print(f"📊 Refined: {sum(len(d['cards']) for d in refined_cards_data)} cards")
            print("💡 Import both decks to compare and choose the best version!")
            
            if progress_file.exists():
                progress_file.unlink()
                print("🧹 Cleaned up progress file")
            
            return [original_apkg, refined_apkg]
        else:
            print("\n📦 Creating Anki package...")
            apkg_path = self.create_anki_package(all_cards_data, lecture_name, images, output_dir)
            
            # Clean up temp media after package creation
            temp_media_dir = Path(output_dir) / "temp_media"
            if temp_media_dir.exists():
                for file in temp_media_dir.glob("*"):
                    file.unlink()
                temp_media_dir.rmdir()
            
            if progress_file.exists():
                progress_file.unlink()
                print("🧹 Cleaned up progress file")
            
            return apkg_path
    
    def process_folder(self, folder_path: str, output_dir: str = "anki_output", resume: bool = True, advanced_mode: bool = False):
        """Process all PDFs in a folder with resume capability."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        print(f"\n📁 Found {len(pdf_files)} PDF files in {folder}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        if advanced_mode:
            print("🧠 Advanced mode: Enabled (will critique and refine cards)")
        
        progress_dir = Path(output_dir) / "progress"
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
    🏥 Ankify : Lecture to Artificially Intelligent Flashcards (Advanced Edition)
    ======================================================
    
    Features:
    - Automatic retry on API errors
    - Resume from interruptions
    - Single/Multiple card modes
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
        print("  --tags=tag1,tag2     Add custom tags to all cards")
        print("  --style=key=value    Custom styling (see examples)")
        print("\nStyle options:")
        print("  --style=background=#f0f0f0,text_color=#333,cloze_color=#0066cc")
        print("  --style=font_family=Georgia,font_size=22px")
        print("\nExamples:")
        print("  python script.py sk-abc... lecture.pdf --single-card --advanced")
        print("  python script.py sk-abc... /lectures/ --tags=cardiology,exam2024")
        print("  python script.py sk-abc... lecture.pdf --style=background=#1a1a1a,text_color=#fff")
        return
    
    api_key = sys.argv[1]
    path = sys.argv[2]
    
    # Parse options
    resume = "--no-resume" not in sys.argv
    single_card_mode = "--single-card" in sys.argv
    advanced_mode = "--advanced" in sys.argv
    
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
    
    generator = MedicalAnkiGenerator(
        api_key, 
        single_card_mode=single_card_mode,
        custom_tags=custom_tags,
        card_style=card_style
    )
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        generator.process_lecture(path, resume=resume, advanced_mode=advanced_mode)
    elif os.path.isdir(path):
        generator.process_folder(path, resume=resume, advanced_mode=advanced_mode)
    else:
        print("❌ Please provide a valid PDF file or folder path")

if __name__ == "__main__":
    main()