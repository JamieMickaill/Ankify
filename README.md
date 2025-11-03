# Ankify: AI-Powered Anki Card Generation from Medical Lectures

<div align="center">

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)
![Anki](https://img.shields.io/badge/Anki-Compatible-red.svg)

**Transform your medical lecture PDFs into high-quality Anki flashcards with advanced AI**

</div>

## 🏥 Overview

Ankify is an intelligent flashcard creation tool designed specifically for medical students. While AnKing cards are excellent for STEP exams, they often don't cover institution-specific lecture content. Ankify bridges this gap by automatically generating comprehensive, exam-ready Anki cards directly from your PDF lecture slides.

### ✨ Key Features

- **🤖 AI-Powered Analysis**: Uses OpenAI's most advanced models (o3) to understand medical content
- **📚 Complete Context**: Every card includes the source slide image and clinical relevance
- **🎯 Medical Student Focus**: Cards emphasize practical knowledge over minutiae
- **🔄 Smart Cloze Deletions**: Intelligently places cloze deletions to maximize learning
- **🧠 Advanced Refinement**: Optional AI critique pass to optimize card quality
- **💾 Resume Capability**: Automatically saves progress - never lose work
- **🎨 Customizable Styling**: Personalize card appearance with custom colors and fonts
- **📦 Batch Processing**: Process entire folders efficiently with compression options
- **🏷️ Automatic Tagging**: Organize cards with slide numbers, topics, and custom tags

## 📋 Requirements

### Python Dependencies
```bash
pip install pymupdf pillow requests genanki
```

### API Requirements
- OpenAI API key with access to o3 model
- Get your key at: https://platform.openai.com/api-keys

## 🚀 Quick Start

### Basic Usage
```bash
python ankify.py YOUR_API_KEY lecture.pdf --multiple-cloze
```

### Process Entire Folder
```bash
python ankify.py YOUR_API_KEY /path/to/lectures/  --multiple-cloze
```

## 🎮 Command Line Options

### Core Options
- `--single-card` - All cloze deletions on one card (uses {{c1::}} only)
- `--no-resume` - Start fresh without resuming from previous progress
- `--test-mode` - Pause before each API call for manual confirmation that the current setup is working
- `--flex-processing` - use cheaper AI API calls at the cost of speed/reliability 

### Performance Options
- `--compress=LEVEL` - Image compression level:
  - `none` - Original quality 
  - `low` - 1024px, JPEG 90%
  - `medium` - 800px, JPEG 85% 
  - `high` - 512px, JPEG 80% (default)

### Customization Options
- `--tags=tag1,tag2` - Add custom tags to all cards
- `--add-hints` - Add descriptive hints to cloze deletions (requires --advanced)
- `--style=key=value` - Custom card styling:
  - `background=#hexcolor` - Background color
  - `text_color=#hexcolor` - Text color
  - `cloze_color=#hexcolor` - Cloze deletion color
  - `bold_color=#hexcolor` - Bold text color
  - `font_family=name` - Font family
  - `font_size=size` - Font size (e.g., 20px)

## 📖 Usage Examples

### Standard Processing
```bash
python ankify.py sk-abc123... cardiology_lecture.pdf  --multiple-cloze
```

### Cost-Efficient Batch Processing
```bash
python ankify.py sk-abc123... /lectures/ --multiple-cloze
```

### Full Featured with Custom Styling
```bash
python ankify.py sk-abc123... lecture.pdf  --multiple-cloze --tags=cardiology,semester2 --style=background=#1a1a1a,text_color=#ffffff,cloze_color=#00ff00 
```

## 📂 Input/Output Structure

### Inputs
- **Single PDF**: Individual lecture file
- **Folder**: Directory containing multiple PDF files
- Supports multi-part lectures (combine parts 1/2/3 for better context)

### Outputs
```
anki_output/
├── LectureName.apkg              # Standard deck
├── LectureName_Original.apkg     # Original cards (advanced mode)
├── LectureName_Refined.apkg      # Refined cards (advanced mode)
├── LectureName_cards_reference.txt # Text reference of all cards
├── progress/                     # Resume data
└── anki_logs/                    # Processing logs
    ├── ankify_log_*.log          # Session logs
    └── refinements/              # Refinement decisions
        ├── *.json                # Detailed refinement data
        └── *.txt                 # Human-readable summary
```

## 💡 Tips for Optimal Results

### Card Quality
- **Use Advanced Models**: Always use o3 or GPT5 for best results
- **Combine Multi-Part Lectures**: Process "Part 1/2/3" as a single file to maintain context
- **Review Generated Cards**: While cards are self-contained, personal review ensures they match your learning style

### Cost Efficiency
- **Batch Processing** and **Compression**: These are implemented by default, cutting huge costs on API calls while retaining quality
- **Combine Lectures**: Merge related PDFs before processing, remove large redundant images
- **Test Mode**: Use `--test-mode` to verify quality before processing large batches

### Troubleshooting
- **Resume from Interruption**: Processing automatically resumes - just run the same command
- **API Rate Limits**: The tool automatically handles rate limiting with exponential backoff
- **Memory Issues**: Use compression for large PDFs or many slides

## 🏗️ How It Works

1. **PDF Extraction**: Converts each slide to a high-quality image
2. **AI Analysis**: Sends slides to OpenAI API for medical content extraction
3. **Card Generation**: Creates cloze deletion cards with clinical context
4. **Refinement** (optional): AI reviews and optimizes all cards via multiple iterative steps -> improving cloze content, card grouping, hints and general quality
5. **Package Creation**: Builds `.apkg` file with cards, images, and styling

## 🎯 Card Quality Features

- **Self-Contained**: Each card includes enough context to be understood independently
- **Clinical Relevance**: Highlights practical applications and clinical pearls
- **Smart Cloze Placement**: Tests key facts without ambiguity
- **Medical Abbreviations**: Automatically expands abbreviations at least once
- **Duplicate Detection**: Removes redundant cards in advanced mode
- **Focus Areas**: Emphasizes pathophysiology, clinical features, treatments, and diagnoses

## 📊 Example Card

```
Text: In type 2 diabetes, {{c1::Metformin}} is the first-line medication because it {{c2::does not cause hypoglycemia}} and has {{c3::cardiovascular benefits}}

Clinical Pearl: 💡 Always check renal function before prescribing

Context: Essential knowledge for diabetes management in primary care

[Slide image attached]
```

## 🔒 Privacy & Security

- All processing is done via secure HTTPS connections to OpenAI
- No data is stored beyond your local progress files
- API keys are never logged or stored

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- GUI interface
- Support for additional AI providers
- Enhanced image preprocessing
- More card format options

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Designed specifically for medical students preparing for exams
- Inspired by the need for institution-specific study materials
- Built to complement existing resources like AnKing

---

<div align="center">
Made with ❤️ for medical students everywhere
</div>
