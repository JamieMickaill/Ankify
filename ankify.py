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
    def __init__(self, openai_api_key: str, single_card_mode: bool = False):
        self.api_key = openai_api_key
        self.single_card_mode = single_card_mode
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
        
        # Create a custom Anki model for cloze cards with images
        model_id = 1234567890  # Fixed ID for consistency
        self.cloze_model = genanki.Model(
            model_id,
            'Medical Cloze with Image',
            fields=[
                {'name': 'Text'},
                {'name': 'Extra'},
            ],
            templates=[
                {
                    'name': 'Cloze',
                    'qfmt': '{{cloze:Text}}<br><br>{{Extra}}',
                    'afmt': '{{cloze:Text}}<br><br>{{Extra}}',
                },
            ],
            css='''
                .card {
                    font-family: arial;
                    font-size: 20px;
                    text-align: center;
                    color: black;
                    background-color: white;
                }
                .cloze {
                    font-weight: bold;
                    color: blue;
                }
                img {
                    max-width: 100%;
                    max-height: 600px;
                    margin-top: 20px;
                }
            ''',
            model_type=genanki.Model.CLOZE
        )
        
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
        
        Example:
        [
          {{
            "text": "{{{{c1::Peristalsis}}}} is the {{{{c2::autonomous rhythmic contraction}}}} of smooth muscle in the GI tract",
            "facts": ["Peristalsis", "autonomous rhythmic contraction"],
            "context": "Key GI physiology concept"
          }},
          {{
            "text": "The normal resting heart rate is {{{{c1::60-100}}}} beats per minute in {{{{c2::adults}}}}",
            "facts": ["60-100", "adults"],
            "context": "Important vital sign parameter"
          }}
        ]
        
        Create as many cards as needed to cover all testable information on this slide. Use multiple cloze deletions in a single card when testing related concepts. **Make the cards as concise as possible while retaining the key points**"""
        
        payload = {
            "model": "o3",  # Updated to o3
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
            "max_completion_tokens": 100000  # Updated for o3
        }
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter
                if attempt > 0:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"\n  ⏳ Retry {attempt}/{max_retries} after {wait_time:.1f}s wait...", end='', flush=True)
                    time.sleep(wait_time)
                
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120  # Increased timeout for o3
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    # Extract JSON from the response
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        cards_data = json.loads(json_match.group())
                        # Convert to single card format if needed
                        if self.single_card_mode:
                            for card in cards_data:
                                card['text'] = self.convert_to_single_card_format(card['text'])
                        return {
                            "page_num": page_num,
                            "cards": cards_data
                        }
                elif response.status_code == 429:  # Rate limit
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"\n  ⚠️ Rate limited. Waiting {wait_time}s...", end='', flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"\n  ❌ API Error: {response.status_code} - {response.text[:100]}...", end='', flush=True)
                    
            except requests.exceptions.Timeout:
                print(f"\n  ⏱️ Request timeout (attempt {attempt + 1}/{max_retries})", end='', flush=True)
            except requests.exceptions.ConnectionError as e:
                print(f"\n  🔌 Connection error (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}...", end='', flush=True)
            except Exception as e:
                print(f"\n  ❗ Error (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}...", end='', flush=True)
        
        print(f"\n  ❌ Failed after {max_retries} attempts", end='', flush=True)
        return {"page_num": page_num, "cards": []}
    
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
    
    def create_anki_package(self, cards_data: List[Dict], lecture_name: str, images: List[Tuple[Image.Image, int]], output_dir: str):
        """Create Anki package (.apkg) with cards and images using genanki."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique deck ID
        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(
            deck_id,
            f'Medical::{lecture_name}'
        )
        
        # Prepare media files list
        media_files = []
        
        # Create a mapping of page numbers to images
        page_to_image = {page_num: img for img, page_num in images}
        
        # For text output
        all_cards_text = []
        card_number = 1
        
        # Create temporary directory for images
        temp_media_dir = output_path / "temp_media"
        temp_media_dir.mkdir(exist_ok=True)
        
        card_mode_text = "Single Card Mode (all blanks shown together)" if self.single_card_mode else "Multiple Card Mode (separate cards for each blank)"
        all_cards_text.append(f"Card Mode: {card_mode_text}")
        all_cards_text.append("-" * 50)
        
        for slide_data in cards_data:
            page_num = slide_data['page_num']
            slide_cards = slide_data['cards']
            
            # Save the slide image
            image_filename = f"slide_{lecture_name}_{page_num:03d}.png"
            if page_num in page_to_image:
                image_path = temp_media_dir / image_filename
                page_to_image[page_num].save(image_path, "PNG", optimize=True)
                media_files.append(str(image_path))
            
            for card in slide_cards:
                # Create the note with cloze text and image
                note_text = card['text']
                extra_content = f'<img src="{image_filename}"><br><br><i>Context: {card.get("context", "")}</i>'
                
                note = genanki.Note(
                    model=self.cloze_model,
                    fields=[note_text, extra_content],
                    tags=[f'slide_{page_num}', lecture_name.replace(" ", "_"), 'medical']
                )
                deck.add_note(note)
                
                # Add to text file
                all_cards_text.append(f"Card {card_number} (Slide {page_num}):")
                all_cards_text.append(f"Text: {note_text}")
                all_cards_text.append(f"Facts tested: {', '.join(card.get('facts', []))}")
                all_cards_text.append(f"Context: {card.get('context', 'N/A')}")
                all_cards_text.append("-" * 50)
                
                card_number += 1
        
        # Create the package
        package = genanki.Package(deck)
        package.media_files = media_files
        
        # Save the .apkg file
        apkg_filename = output_path / f"{lecture_name}.apkg"
        package.write_to_file(str(apkg_filename))
        
        # Save cards text file for reference
        text_file = output_path / f"{lecture_name}_cards_reference.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Anki Cards for {lecture_name}\n")
            f.write(f"Total cards: {card_number - 1}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(all_cards_text))
        
        # Clean up temporary media directory
        for file in temp_media_dir.glob("*"):
            file.unlink()
        temp_media_dir.rmdir()
        
        print(f"\n✅ Successfully created {card_number - 1} flashcards")
        print(f"📦 Anki package saved: {apkg_filename}")
        print(f"📄 Reference text saved: {text_file}")
        
        return apkg_filename
    
    def process_lecture(self, pdf_path: str, output_dir: str = "anki_output", resume: bool = True):
        """Process a single lecture PDF with resume capability."""
        lecture_name = Path(pdf_path).stem
        print(f"\n🔍 Processing lecture: {lecture_name}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        
        # Create progress file path
        progress_dir = Path(output_dir) / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_file = progress_dir / f"{lecture_name}_progress.pkl"
        
        # Check for existing progress
        progress_data = None
        if resume:
            progress_data = self.load_progress(progress_file)
            if progress_data:
                print(f"📂 Found existing progress: {len(progress_data['completed_slides'])} slides already processed")
        
        # Convert PDF to images
        print("📄 Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)
        print(f"✅ Extracted {len(images)} slides")
        
        # Initialize or load progress
        if progress_data is None:
            progress_data = {
                'lecture_name': lecture_name,
                'total_slides': len(images),
                'completed_slides': [],
                'cards_data': [],
                'start_time': datetime.now().isoformat(),
                'single_card_mode': self.single_card_mode
            }
        
        # Analyze each slide
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
            
            # Update progress
            completed_slides.add(page_num)
            progress_data['completed_slides'] = list(completed_slides)
            progress_data['cards_data'] = all_cards_data
            progress_data['last_update'] = datetime.now().isoformat()
            
            # Save progress after each slide
            self.save_progress(progress_file, progress_data)
        
        # Create Anki package
        print("\n📦 Creating Anki package...")
        apkg_path = self.create_anki_package(all_cards_data, lecture_name, images, output_dir)
        
        # Clean up progress file on successful completion
        if progress_file.exists():
            progress_file.unlink()
            print("🧹 Cleaned up progress file")
        
        return apkg_path
    
    def process_folder(self, folder_path: str, output_dir: str = "anki_output", resume: bool = True):
        """Process all PDFs in a folder with resume capability."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        print(f"\n📁 Found {len(pdf_files)} PDF files in {folder}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        
        # Check for overall progress
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
                self.process_lecture(str(pdf_file), output_dir, resume=resume)
                successful += 1
                
                # Update folder progress
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
        
        # Clean up folder progress file if all completed
        if successful == len(pdf_files) and folder_progress_file.exists():
            folder_progress_file.unlink()
            print("🧹 Cleaned up folder progress file")

# Example usage and setup instructions
def main():
    print("""
    🏥 Medical Lecture to Anki Converter (with Resume Support)
    =========================================================
    
    This tool converts medical lecture PDFs into Anki flashcards using AI.
    Now with automatic retry, resume capabilities, and cloze mode options!
    
    Features:
    - Automatic retry on API errors (up to 5 attempts)
    - Resume from where you left off if interrupted
    - Progress tracking for each slide
    - Exponential backoff for rate limits
    - Single card mode: All clozes on one card ({{c1::}} only)
    - Multiple card mode: Separate cards for each cloze ({{c1::}}, {{c2::}}, etc.)
    
    Requirements:
    1. Install required packages:
       pip install pymupdf pillow requests genanki
    
    2. Get your OpenAI API key from https://platform.openai.com/api-keys
    
    3. Run the script with your API key
    """)
    
    # Check if running as script
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <api_key> <pdf_file_or_folder> [options]")
        print("\nOptions:")
        print("  --single-card    Create single cards with all blanks (all {{c1::}})")
        print("  --no-resume      Don't resume from previous progress")
        print("\nExamples:")
        print("  python script.py sk-abc123... cardiology_lecture.pdf")
        print("  python script.py sk-abc123... /path/to/lectures_folder/")
        print("  python script.py sk-abc123... lecture.pdf --single-card")
        print("  python script.py sk-abc123... lecture.pdf --no-resume --single-card")
        return
    
    api_key = sys.argv[1]
    path = sys.argv[2]
    
    # Parse options
    resume = "--no-resume" not in sys.argv
    single_card_mode = "--single-card" in sys.argv
    
    generator = MedicalAnkiGenerator(api_key, single_card_mode=single_card_mode)
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        generator.process_lecture(path, resume=resume)
    elif os.path.isdir(path):
        generator.process_folder(path, resume=resume)
    else:
        print("❌ Please provide a valid PDF file or folder path")

if __name__ == "__main__":
    main()