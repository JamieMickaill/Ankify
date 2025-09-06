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
        """Get good vs bad cloze examples - UPDATED WITH NEW EXAMPLES."""
        return """
GOOD vs BAD CLOZE EXAMPLES:

✅ GOOD: "The IVC is formed by the junction of the {{c1::left and right common iliac veins}}"
❌ BAD: "The IVC is {{c1::formed by the junction of the left and right common iliac veins}}" (too vague)

✅ GOOD: "Malignant pericardial effusion should not contain {{c1::malignant}} cells"
❌ BAD: "The fluid shouldn't contain {{c1::malignant}} cells" (which fluid?)

✅ GOOD: "Lung cancer prognosis is poor when {{c1::T cells}} are {{c2::inactivated}}"
❌ BAD: "Lung cancer prognosis is poor when {{c1::T cells are inactivated}}" (tests too much at once)

ADDITIONAL EXAMPLES TO AVOID:

❌ BAD: "Visual warning signs of possible breast cancer include skin changes, dimpling, a pulled-in nipple, redness/rash, abnormal nipple discharge, and a visible {{c1::lump}}."
→ Problem: Only tests one symptom when all are important
✅ BETTER: "Visual warning signs of possible breast cancer include {{c1::skin}} changes, {{c1::dimpling}}, a {{c1::pulled-in}} nipple, {{c1::redness/rash}}, abnormal nipple {{c1::discharge}}, and a visible {{c1::lump}}."

❌ BAD: "Future ADC research priorities include optimal sequencing, combination with immunotherapy, biomarker development and understanding {{c1::resistance mechanisms}}."
→ Problem: Future research priorities are not assessable content
✅ SKIP THIS CONTENT - not relevant for medical student assessment

❌ BAD: "Multiple DESTINY-Breast trials are evaluating T-DXd in {{c1::earlier-stage (neoadjuvant and adjuvant)}} HER2-positive disease"
→ Problem: Unexplained abbreviation "T-DXd" makes card not self-contained
✅ BETTER: "{{c1::Trastuzumab deruxtecan (T-DXd)}} trials are evaluating use in {{c2::earlier-stage}} HER2-positive disease"

❌ INCOMPLETE: "{{c1::CDK4/6 inhibitors}} (e.g., palbociclib) block the {{c1::G1→S transition}} of the cell cycle"
→ Problem: Misses the drug name and target population
✅ COMPLETE: "{{c1::CDK4/6 inhibitors}} like {{c2::palbociclib}} block {{c3::G1→S transition}} in {{c4::ER-positive}} breast cancer cells"

❌ BAD: "Gastrointestinal side effects such as {{c1::nausea}} and {{c1::diarrhoea}} frequently accompany ADC treatment."
→ Problem: No context - what is ADC?
✅ BETTER: "Gastrointestinal side effects such as {{c1::nausea}} and {{c1::diarrhoea}} frequently accompany {{c1::antibody-drug conjugate (ADC)}} treatment"

❌ BAD: "When pCR is not achieved after neoadjuvant therapy, escalation may involve adding {{c1::capecitabine}}"
→ Problem: Unexplained abbreviation "pCR"
✅ BETTER: "When {{c1::pathological complete response (pCR)}} is not achieved, escalation may involve adding {{c2::capecitabine}}"

❌ ILLOGICAL: "Because external-beam breast RT does {{c1::not make patients radioactive}}, they can {{c1::safely drive themselves}}"
→ Problem: Creates illogical connection between radioactivity and driving
✅ SKIP or REWORD to focus on the actual safety aspect

❌ BAD: "Unless ER-positive tumour is <10 mm, offer {{c1::5 years of endocrine therapy}}; use {{c1::tamoxifen for pre-menopausal}}"
→ Problem: Too much info in second cloze
✅ BETTER: "For ER-positive tumours ≥10 mm, offer {{c1::5 years}} of endocrine therapy; use {{c2::tamoxifen}} for {{c3::pre-menopausal}} women"

❌ BAD: "Adding a {{c1::CDK4/6 inhibitor}} to an {{c1::aromatase inhibitor}} roughly {{c2::doubles progression-free survival}} versus AI alone"
→ Problem: "AI alone" reveals second cloze deletion
✅ BETTER: "Adding {{c1::CDK4/6 inhibitor}} to {{c2::aromatase inhibitor}} {{c3::doubles}} progression-free survival in metastatic ER+ cancer"

❌ BAD: "At presentation, ~{{c1::95%}} of TNBC cases are {{c1::early-stage}}, whereas only 5% are metastatic"
→ Problem: "5% are metastatic" reveals the 95% figure
✅ BETTER: "At presentation, most TNBC cases are {{c1::early-stage (I-III)}} rather than {{c1::metastatic (stage IV)}}"

❌ TOO VAGUE: "Adjuvant systemic therapy provides lifelong improvement in {{c1::recurrence-free and overall survival}}"
→ Problem: Too ambiguous - what specific outcomes?
✅ BETTER: "Adjuvant systemic therapy improves {{c1::recurrence-free}} and {{c2::overall}} survival"

❌ INCOMPLETE: "G-CSF after chemotherapy reduces the {{c1::depth and duration of neutropenia}}"
→ Problem: Drug name (G-CSF) is more important than effect details
✅ BETTER: "{{c1::Granulocyte colony-stimulating factor (G-CSF)}} reduces {{c2::neutropenia}} after chemotherapy"

❌ OBVIOUS: "Tubular, cribriform and mucinous carcinomas are examples of {{c1::special histologic subtypes}}"
→ Problem: Obviously they're histologic subtypes
✅ BETTER: "{{c1::Tubular}}, {{c1::cribriform}} and {{c1::mucinous}} carcinomas have {{c2::better prognosis}} than ductal NST"

❌ POOR CHOICE: "Systematic breast palpation begins with light pressure, followed by medium and deep pressure before {{c1::moving to the next zone}}"
→ Problem: Wrong focus - moving zones is obvious
✅ BETTER: "Systematic breast palpation uses {{c1::light}}, then {{c2::medium}}, then {{c3::deep}} pressure"

❌ NO CONTEXT: "Randomised trials show lower {{c1::mastectomy rate}} when neoadjuvant therapy is used"
→ Problem: Needs breast cancer context
✅ BETTER: "In breast cancer, neoadjuvant therapy reduces {{c1::mastectomy}} rates by enabling {{c2::breast-conserving surgery}}

❌ BAD: "Pain is usually the {{c1::first symptom::clinical feature}} of malignant spinal cord compression and typically precedes neurological deficits by a median of {{c1::about 7 weeks::duration}}."
→ Problem: The actual symptom "pain" should be tested, not that it's "first"
✅ BETTER: "{{c1::Pain}} is usually the first symptom of malignant spinal cord compression, preceding neurological deficits by {{c2::7 weeks}}"

❌ BAD: "In prostate cancer, {{c1::androgen-deprivation therapy (ADT)::therapy type}} is the {{c1::mainstay systemic treatment::treatment role}}, used in {{c2::both curative and advanced::disease setting}} disease settings."
→ Problem: Testing "mainstay" is meaningless; also redundant cloze hints
✅ BETTER: "In prostate cancer, {{c1::androgen-deprivation therapy (ADT)}} is used in both {{c2::curative}} and {{c2::advanced}} disease settings"

❌ BAD: "Cytotoxic chemotherapy in prostate cancer is generally reserved for the {{c1::metastatic::stage}} setting rather than for localised disease."
→ Problem: If not localised, obviously advanced/metastatic
✅ BETTER: "In prostate cancer, {{c1::cytotoxic chemotherapy}} is reserved for {{c2::metastatic}} disease"

❌ BAD: "Most international guidelines {{c1::do not recommend routine population PSA screening::recommendation}} and instead emphasise {{c1::shared decision-making based on informed choice::approach}}."
→ Problem: Extremely ambiguous without PSA context
✅ BETTER: "Most international guidelines {{c1::do not recommend}} routine population {{c2::PSA}} screening, emphasising {{c3::shared decision-making}}"

❌ BAD: "In AJCC 2017 staging, advancing primary tumour category from T1 to T3b/T4 produces progressively worse recurrence-free survival following surgery."
→ Problem: No reference to which cancer (acinar prostatic adenocarcinoma)

❌ BAD: "Under the 2023 eviQ guidelines, men with {{c1::castrate-resistant metastatic prostate cancer (mCRPC)::indication}} should commence genetic work-up with {{c1::somatic (tumour) testing::investigation}}, regardless of personal or family history."
→ Problem: When cloze deleted, too ambiguous - no hint about which disease

❌ BAD: "Sickled red cells have {{c1::decreased deformability and increased endothelial adhesion::cellular properties}}, leading to {{c1::microvascular obstruction with hypoxia, acidosis and coagulation activation::consequence}}."
→ Problem: Deletions way too large
✅ BETTER: "Sickled red cells have decreased {{c1::deformability}} and increased {{c2::endothelial adhesion}}, causing {{c3::microvascular obstruction}} with {{c4::hypoxia}}, {{c4::acidosis}} and {{c4::coagulation activation}}"

❌ BAD: "Haemoglobin chromatography separates Hb variants and confirms that {{c1::Hb S homozygosity causes sickle cell disease}}"
→ Problem: Can confirm countless things - too ambiguous
✅ BETTER: "Haemoglobin chromatography confirms that Hb {{c1::S::type}} {{c1::homozygosity::inheritance}} causes {{c2::sickle cell disease}}"

❌ BAD: "In α-thalassaemia, excess unpaired {{c1::β chains form HbH (β₄)::pathophysiology}} and excess {{c1::γ chains form Hb Barts (γ₄)::pathophysiology}}."
→ Problem: Cloze deletions too large and ambiguous
✅ BETTER: "In α-thalassaemia, excess unpaired {{c1::β}} chains form {{c2::HbH (β₄)}} and excess {{c3::γ}} chains form {{c4::Hb Barts (γ₄)}}"

❌ BAD: "Long-term antiplatelet prophylaxis blocks amplification pathways: aspirin irreversibly inhibits {{c1::thromboxane synthesis::mechanism}}, whereas drugs like {{c1::clopidogrel::drug}} block the platelet {{c1::ADP receptor::receptor}}; this results in incomplete platelet inhibition but a wide safety margin suitable for chronic prevention of arterial thrombosis"
→ Problem: Missed key fact: drug name "Aspirin".
✅ BETTER: "Long-term antiplatelet prophylaxis blocks amplification pathways: {{c2::aspirin}} irreversibly inhibits {{c1::thromboxane synthesis::mechanism}}, whereas drugs like {{c2::clopidogrel::drug}} block the platelet {{c1::ADP receptor::receptor}}; this results in incomplete platelet inhibition but a wide safety margin suitable for chronic prevention of arterial thrombosis."

❌ BAD: "Plasma {{c1::fibrinogen}} is both an {{c1::abundant}} protein and an {{c1::acute-phase reactant}}, rising markedly after injury or inflammation."
→ Problem: The fact that fibrinogen is "Abundant" is not important, this is not a testable concept for a medical student
✅ BETTER: "Plasma {{c2::fibrinogen}} is both an abundant protein and an {{c1::acute-phase reactant}}, rising markedly after injury or inflammation."

❌ BAD: "The first {{c1::management}} step when an acute haemolytic transfusion reaction is suspected is to {{c1::stop the transfusion immediately}}."
→ Problem: The fact that a step is a "management" step is not something a student would need to memorize. Doesn't test the student on the reaction type.
✅ BETTER: "The first management step when an {{c1::acute haemolytic transfusion}} reaction is suspected is to {{c2::stop the transfusion immediately}}."


CRITICAL CARD EVALUATION QUESTIONS:
Before finalizing any card, ask:
1. Once cloze sections are deleted, can you easily guess what this card is about (the main topic - which disease, drug, cell, treatment)?
2. If not, the card is too ambiguous. Either:
   - Add a hint to remove ambiguity
   - Add context within the sentence to remove ambiguity
3. Are all important facts being tested, not just one arbitrary detail?

"""
    
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
✅ COMPLETE: "{{c1::Early}} breast cancer is confined to {{c2::breast ± axillary nodes}} and is {{c3::potentially curable}}"
→ Tests both type, definition and prognosis

❌ SUPERFICIAL: "{{c1::Palbociclib}} blocks G1-to-S transition"
→ Only tests drug name
✅ COMPREHENSIVE: "{{c1::CDK4/6 inhibitors}} like {{c2::palbociclib}} block {{c3::G1-to-S phase transition}}, arresting {{c4::proliferation}} of ER+ cells"
→ Tests drug class, example, mechanism, and effect

CRITICAL REMINDERS:
- Test ALL important facts in a statement, not just one
- Ensure abbreviations are spelled out at least once
- Provide sufficient context for self-contained understanding
- Focus on clinically relevant information, not research priorities
- Avoid cards where remaining context reveals the cloze deletion"""
    
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
   - Future research priorities or ongoing trials

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
   - Avoids testing exact percentages unless clinically critical (prefer testing the condition/gene/drug name or using "common/rare")
   - Spells out all abbreviations at least once
   - Includes all important symptoms/signs when listing them"""

    @staticmethod
    def get_fine_tuned_cloze_principles() -> str:
        """Get fine-tuned cloze grouping principles."""
        return """
FINE-TUNED CLOZE GROUPING PRINCIPLES:

GOAL: Create the FEWEST separate cloze cards possible while preventing excessive ambiguity.

CORE PRINCIPLE: Group related concepts together with the same cloze number. When in doubt, use fewer cloze numbers rather than more (it's easier to manually split than to combine).

GROUPING GUIDELINES:

1. ALWAYS GROUP TOGETHER (same cloze number):
   - Multiple symptoms/signs of the same condition
   - Multiple examples of the same category
   - Related anatomical structures in a list
   - Multiple drugs with the same indication
   - Related lab values or ranges
   - Components of a syndrome or criteria

2. SEPARATE INTO DIFFERENT CLOZE NUMBERS:
   - Test name vs test result
   - Drug name vs mechanism of action
   - Disease vs treatment
   - Cause vs effect
   - Structure vs function
   - Normal vs abnormal values

3. EXAMPLES OF GOOD FINE-TUNED GROUPING:

✅ GOOD: "Symptoms of DKA include {{c1::polyuria}}, {{c1::polydipsia}}, and {{c1::weight loss}}, with labs showing {{c2::glucose >250}} and {{c2::pH <7.3}}"
→ Groups symptoms together (c1) and lab values together (c2)

✅ GOOD: "{{c1::Metformin}} and {{c1::sulfonylureas}} are oral agents, while {{c2::insulin}} is injectable for diabetes"
→ Groups oral agents together, separates by route

✅ GOOD: "The {{c1::indirect antiglobulin test (IAT)}} detects {{c2::unexpected red-cell antibodies}}"
→ Separates test from what it detects to avoid ambiguity

✅ GOOD: "{{c2::Aspirin}} inhibits {{c1::thromboxane synthesis}}, while {{c2::clopidogrel}} blocks {{c1::ADP receptor}}"
→ Groups drugs together and mechanisms together

4. EXAMPLES TO AVOID:

❌ BAD: "The antibody screen looks for {{c1::unexpected antibodies}} using {{c1::IAT}}"
→ Too ambiguous when both are hidden - separate test from target

❌ BAD: "DKA presents with {{c1::polyuria}}, {{c2::polydipsia}}, {{c3::weight loss}}"
→ Unnecessary separation of related symptoms

5. DECISION FRAMEWORK:
   - Can the blanks be logically filled when hidden together? → Use same cloze
   - Would hiding them together make the card unanswerable? → Use different cloze
   - Are they examples of the same category? → Use same cloze
   - Do they represent cause and effect? → Use different cloze

6. ERR ON THE SIDE OF GROUPING:
   When uncertain, prefer fewer cloze numbers. It's easier to manually create additional cards later than to merge multiple cards.

REMEMBER: The goal is intelligent grouping that creates fewer cards while maintaining the ability to answer each card without excessive ambiguity."""
    


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

    @staticmethod
    def get_hint_instructions() -> str:
        """Get instructions for adding cloze hints in critique mode."""
        return """
CLOZE HINT GUIDELINES:
When refining cards, add descriptive category hints to cloze deletions to narrow down guessing without giving away the answer.
!! IMPORTANT: ONLY INCLUDE HINTS IF THEY REDUCE AMBIGUITY !! 
!! IMPORTANT: DO NOT INCLUDE HINTS WITH NO HINT VALUE - I.E. NO REDUCTION IN AMBIGUITY!!

FORMAT: {{c1::answer::hint}}

GOOD HINT CATEGORIES:
- "drug/medication" - for drug names
- "dose/duration" - for dosing or time periods
- "receptor/marker" - for biological markers
- "stage/grade" - for staging or grading
- "percentage/rate" - for statistics
- "mechanism" - for pathophysiology
- "clinical feature" - for signs/symptoms
- "investigation" - for diagnostic tests
- "hormonal status" - for pre/post-menopausal
- "receptor status" - for ER/PR/HER2 or other receptors
- "side effect" - for adverse effects
- "contraindication" - for when not to use
- "indication" - for when to use
- "prostate cancer type" - for specific prostate cancer variants

EXAMPLES:
✅ "Tamoxifen is preferred for {{c1::premenopausal::hormonal status}} women"
Why: Initially broad category of types of women is narrowed down to those of different hormonal status. Does not give away answer.
✅ "TNBC shows {{c1::ER < 1%, PR < 1%::receptor expression}}"
Why: TNBC could show anything (initially ambiguous), hint narrowed down to those of different receptor expression. Does not give away answer.
✅ "Adjuvant therapy is given for {{c1::5 years::duration}} (10 if high-risk)"
Why: Initial interpretation could be indication, demographic, pathological subtype -> hint narrows down to duration without giving away answer.
✅ "{{c1::CDK4/6 inhibitors::drug class}} like palbociclib block cell cycle"

AVOID:
❌ Valueless hints: The prostate cancer survivorship essentials framework lists six domains: {{c1::Health promotion &amp; advocacy::domain}}, {{c1::Evidence‑based interventions::domain}}, {{c1::Personal agency::domain}}, {{c1::Vigilance::domain}}, {{c1::Care coordination::domain}}, and {{c1::Shared management::domain}}.
WHY: Domain does not give any hint at all, "domain" is a generic placeholder for anything and is included in the cloze already. NO REDUCTION IN AMBIGUITY

❌ Generic placeholders: Priority actions for men with prostate cancer include enhancing {{c1::patient‑clinician communication::action}}, developing a {{c1::survivorship toolkit::action}}, expanding {{c1::multi‑modal care::action}}, reducing {{c1::out‑of‑pocket costs::action}}, promoting {{c1::exercise::action}}, harnessing {{c1::technology for access::action}}, and building {{c1::specialist outreach capacity::action}}.
WHY: "action" is a generic placeholder that provides no hint value and is already mentioned in the cloze

❌ Inaccurate hints: Whole blood is roughly {{c1::55 % plasma::percentage}} and {{c1::45 % formed elements (red cells, white cells, platelets)::percentage}}.
WHY: The deletion is not just a percentage but a percentage AND a component of blood. Hints must accurately describe the WHOLE cloze deletion content

❌ Redundant hints: {{c1::nausea::nausea symptom}}
❌ Hints that give away the answer 

OVERARCHING GOAL: Specific hints without giving away the answer are optimal. This is challenging however, and you should err on the side of ambiguity rather than specificity."""



class MedicalAnkiGenerator:
    def __init__(self, openai_api_key: str, single_card_mode: bool = True,  # Changed default to True
                custom_tags: Optional[List[str]] = None, card_style: Optional[Dict] = None,
                compression_level: str = "high",  # Changed default to "high"
                test_mode: bool = False, 
                add_hints: bool = True):  # Changed default to True, removed batch_mode and preserve_quality
        self.api_key = openai_api_key
        self.single_card_mode = single_card_mode
        self.custom_tags = custom_tags or []
        self.card_style = card_style or {}
        self.compression_level = compression_level
        self.test_mode = test_mode
        self.add_hints = add_hints
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
        self.logger.info(f"Configuration: single_card={self.single_card_mode}, "
                        f"compression={self.compression_level}, test_mode={self.test_mode}, "
                        f"add_hints={self.add_hints}")
        
    def _create_styled_model(self):
        """Create Anki model with custom styling."""
        # Default style values
        bg_color = self.card_style.get('background', 'white')
        text_color = self.card_style.get('text_color', 'black')
        cloze_color = self.card_style.get('cloze_color', 'blue')
        bold_color = self.card_style.get('bold_color', None)  # New option
        font_family = self.card_style.get('font_family', 'arial')
        font_size = self.card_style.get('font_size', '20px')

        # If no bold color specified, use brightened cloze color as before
        if bold_color is None:
            bold_color = self._adjust_color_brightness(cloze_color, 2)
        
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
                color: {bold_color};
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
        
        # Always use high DPI for extraction
        extraction_dpi = 300
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(extraction_dpi/72, extraction_dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append((img, page_num + 1))
        
        doc.close()
        return images

    def image_to_base64(self, image: Image.Image, for_api: bool = True) -> str:
        """Convert PIL Image to base64 string with optional compression.
            Args:
                image: PIL Image to convert
                for_api: If True, apply compression for API calls. If False, preserve quality.
        """
        buffered = io.BytesIO()
        
        if for_api and self.compression_level != "none":
            # Apply compression for API calls
            settings = self.compression_settings[self.compression_level]
            max_size = settings["max_size"]
            quality = settings["quality"]
            img_format = settings["format"]
            
            # Create a copy to avoid modifying the original
            img_copy = image.copy()
            
            if max(img_copy.size) > max_size:
                img_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            if img_format == "JPEG" and img_copy.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', img_copy.size, (255, 255, 255))
                if img_copy.mode == 'P':
                    img_copy = img_copy.convert('RGBA')
                rgb_image.paste(img_copy, mask=img_copy.split()[-1] if img_copy.mode == 'RGBA' else None)
                img_copy = rgb_image
            
            try:
                img_copy.save(buffered, format=img_format, quality=quality, optimize=True)
            except Exception as e:
                print(f"⚠️ Compression failed, using original: {str(e)}")
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
        else:
            # For Anki cards - always preserve quality
            image.save(buffered, format="PNG", optimize=False)
        
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
- Future research priorities or ongoing trials

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
- Spells out all abbreviations at least once
- Includes all important symptoms/signs when listing them

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
    

    def analyze_slides_batch(self, images: List[Tuple[Image.Image, int]], lecture_name: str, max_retries: int = 3) -> List[Dict]:
        """Send multiple slides to OpenAI API in a single batch request."""
        print(f"\n🔄 Batch processing {len(images)} slides in a single API call...")
        self.logger.info(f"Starting batch processing of {len(images)} slides")
        
        # Test mode check
        if self.test_mode:
            print("\n🔍 Ready to batch analyze slides")
            print("Press Enter to continue (or 'quit' to exit)...")
            user_input = input().strip().lower()
            if user_input == 'quit':
                print("👋 Exiting test mode")
                sys.exit(0)
        
        # Prepare all images
        slides_data = []
        for img, page_num in images:
            base64_image = self.image_to_base64(img, for_api=True)  # <-- Changed
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
            "model": "gpt-5",
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 100000,
            "reasoning_effort": "high",
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
                    timeout=600
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
        if self.add_hints:
            print("💡 Hint mode enabled - adding descriptive hints to cloze deletions")
        self.logger.info(f"Starting critique for lecture: {lecture_name}")
        
        # Test mode check
        if self.test_mode:
            print("\n🔍 Ready to critique and refine cards")
            print("Press Enter to continue (or 'skip' to skip refinement, 'quit' to exit)...")
            user_input = input().strip().lower()
            if user_input == 'skip':
                print("⏭️ Skipping refinement")
                return all_cards_data
            elif user_input == 'quit':
                print("👋 Exiting test mode")
                sys.exit(0)
        
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
            "model": "gpt-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 100000,
            "reasoning_effort": "high",
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 30 * attempt  # Wait 30, 60, 90 seconds
                    print(f"\n⏳ Retry {attempt}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=600
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
                    
            except requests.exceptions.Timeout:
                print(f"\n⏱️ Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    print("❌ Refinement failed after all retries")
                    print("⚠️ Using original cards without refinement")
                    return all_cards_data
            except Exception as e:
                print(f"\n❗ Error during refinement: {str(e)}")
                import traceback
                traceback.print_exc()
                if attempt == max_retries - 1:
                    print("⚠️ Using original cards without refinement")
                    return all_cards_data
                
    def _build_critique_prompt(self, lecture_name: str, cards_for_review: List[Dict]) -> str:
        """Build the critique prompt using centralized templates."""
        cloze_format_instruction = f"{PromptTemplates.get_fine_tuned_cloze_principles()}" if not self.single_card_mode else "using ONLY {{c1::}} for all clozes"
        hint_section = ""
        
        if self.add_hints:
            hint_section = f"""
ADDITIONALLY, YOU MUST ADD CLOZE HINTS:
{PromptTemplates.get_hint_instructions()}

When refining cards, add appropriate hints to help students without giving away the answer."""
        
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
7. Ensure ALL key facts are tested (definitions, mechanisms, effects, durations, drugs, indications, interactions)
8. Avoid trivial clozes - test meaningful medical knowledge
9. Avoid testing exact percentages unless clinically critical - prefer testing the subject (gene/condition name) or using descriptors like "common/rare/most common"
10. ENSURE all abbreviations are spelled out at least once
11. REMOVE DUPLICATE CARDS - if multiple cards test the same concept, keep only the best one
12. IF INSTRUCTED TO BOTH ADD HINTS AND REFINE CLOZE GROUPING, ENSURE THAT BOTH TASKS ARE COMPLETED

{hint_section}

Review these {len(cards_for_review)} cloze deletion flashcards and optimize them by:

1. MAINTAINING good cloze patterns - don't make them more ambiguous
2. ENSURING each card remains self-contained and understandable
3. FIXING only genuinely ambiguous cloze deletions
4. ADDING context where needed rather than hiding information
5. MERGING only truly redundant cards
6. REMOVING duplicates that test the same concepts
7. REMOVING only:
   - Cards about titles, objectives, or outlines themselves
   - Cards with unfixable ambiguity
   - Overly specialized technical details
   - Specific research study names/authors (unless landmark)
   - Exact dosing/technical parameters (unless critical safety info)
   - Research methodology minutiae
   - Historical trivia without clinical relevance
   - Subspecialty procedural minutiae
   - Future research priorities or ongoing trials
8. PRESERVING:
   - Specific, focused cloze deletions
   - Clear context around clozes
   - Self-contained card structure
   - Learning outcomes focus
9. EMPHASIZING:
   - Clinical reasoning and decision-making
   - Comparative effectiveness (why choose A over B)
   - Practical applications in general practice
   - Key safety considerations
   - First-line approaches
   - "Why" and "when" rather than exact numbers
   - Clinical decision pathways
10. ADDING clinical pearls that help with real patient care
11. ENSURING medical accuracy while keeping appropriate depth
12. ADDING descriptive hints if requested (without giving away answers)
13. Fine-tuning cloze grouping if requested (without forgetting to also add hints)

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
    }},
    {{
      "action": "merged",
      "original_indices": [5, 8],
      "original_texts": ["Card about CDK4/6 inhibitors", "Another card about CDK4/6 inhibitors"],
      "reason": "Both cards tested the same concept about CDK4/6 inhibitors"
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
        if self.add_hints:
            all_cards_text.append("Hint Mode: Enabled")
        all_cards_text.append("-" * 50)
        
        for slide_data in cards_data:
            page_num = slide_data['page_num']
            slide_cards = slide_data['cards']
            
            image_filename = f"slide_{lecture_name}_{page_num:03d}.png"
            if page_num in page_to_image:
                image_path = temp_media_dir / image_filename
                if not image_path.exists():
                    # Always save at full quality for Anki cards
                    page_to_image[page_num].save(image_path, "PNG", optimize=False)
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
            f.write("Image Quality: Original (preserved)\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(all_cards_text))
        
        print(f"\n✅ Successfully created {card_number - 1} flashcards")
        if self.card_style:
            print(f"🎨 Custom styling applied: {', '.join([f'{k}={v}' for k, v in self.card_style.items()])}")
        print("🖼️ Original image quality preserved in Anki cards")
        print(f"📦 Anki package saved: {apkg_filename}")
        print(f"📄 Reference text saved: {text_file}")
        
        return apkg_filename
    def process_lecture(self, pdf_path: str, output_dir: str = "anki_output", resume: bool = True, 
                    budget_mode: bool = False):  # Changed from advanced_mode to budget_mode
        """Process a single lecture PDF with resume capability."""
        lecture_name = Path(pdf_path).stem
        print(f"\n🔍 Processing lecture: {lecture_name}")
        self.logger.info(f"Processing lecture: {lecture_name} from {pdf_path}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        if budget_mode:
            print("💰 Budget mode: Enabled (skipping refinement stage)")
        else:
            print("🧠 Advanced refinement: Enabled")
        
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
        
        # Always use batch processing
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
                print("⚠️ Batch processing failed")
                return None
        
        # Handle budget mode vs normal (advanced) mode
        if budget_mode and all_cards_data:
            # Budget mode - only create original deck
            print("\n📦 Creating deck (budget mode - no refinement)...")
            apkg_path = self.create_anki_package(all_cards_data, lecture_name, images, output_dir)
            self._cleanup_temp_files(output_dir, progress_file)
            return apkg_path
        else:
            # Normal mode - refine and create only refined deck
            if all_cards_data:
                refined_cards_data = self.critique_and_refine_cards(all_cards_data, lecture_name)
                
                print("\n📦 Creating refined deck...")
                refined_apkg = self.create_anki_package(refined_cards_data, lecture_name, images, output_dir)
                
                self._cleanup_temp_files(output_dir, progress_file)
                
                print("\n🎯 Processing complete!")
                print(f"📊 Original: {sum(len(d['cards']) for d in all_cards_data)} cards")
                print(f"📊 Refined: {sum(len(d['cards']) for d in refined_cards_data)} cards")
                
                return refined_apkg
            else:
                print("⚠️ No cards generated")
                return None

    
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
                  budget_mode: bool = False):  # Changed from advanced_mode to budget_mode
        """Process all PDFs in a folder with resume capability."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        print(f"\n📁 Found {len(pdf_files)} PDF files in {folder}")
        print(f"🎯 Card mode: {'Single card (all blanks together)' if self.single_card_mode else 'Multiple cards (separate blanks)'}")
        if budget_mode:
            print("💰 Budget mode: Enabled (skipping refinement stage)")
        else:
            print("🧠 Advanced refinement: Enabled")
        
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
                self.process_lecture(str(pdf_file), output_dir, resume=resume, budget_mode=budget_mode)
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
    🏥 Ankify: Artificially Intelligent Flashcard Creation
    ======================================================
    
    Features:
    - AI-powered critique and refinement (default)
    - Single card mode (default) with all blanks together
    - Batch processing with smart compression
    - Descriptive hints for better learning (default)
    - Resume from interruptions
    - Custom styling and tags
    - Bold formatting for key terms
    
    Requirements:
    1. Install: pip install pymupdf pillow requests genanki
    2. Get OpenAI API key from https://platform.openai.com/api-keys
    """)
    
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <api_key> <pdf_file_or_folder> [options]")
        print("\nOptions:")
        print("  --budget             Skip refinement stage (faster, cheaper)")
        print("  --multiple-cloze     Intelligent cloze allocation (i.e. grouping by concepts)")
        print("  --no-resume          Start fresh (don't resume)")
        print("  --no-hints           Disable descriptive hints")
        print("  --compress=LEVEL     Image compression (none/low/medium/high) [default: high]")
        print("  --tags=tag1,tag2     Add custom tags to all cards")
        print("  --style=key=value    Custom styling (see examples)")
        print("  --test-mode          Require Enter key before each API call")
        print("\nStyle options:")
        print("  background=#hexcolor    Background color")
        print("  text_color=#hexcolor    Main text color")
        print("  cloze_color=#hexcolor   Cloze deletion color")
        print("  bold_color=#hexcolor    Bold text color")
        print("  font_family=name        Font family")
        print("  font_size=size          Font size (e.g., 20px)")
        print("\nExamples:")
        print("  # Standard processing (with refinement)")
        print("  python script.py sk-abc... lecture.pdf")
        print("\n  # Budget mode (no refinement)")
        print("  python script.py sk-abc... lecture.pdf --budget")
        print("\n  # Multiple cloze cards")
        print("  python script.py sk-abc... lecture.pdf --multiple-cloze")
        print("\n  # Process folder with custom tags")
        print("  python script.py sk-abc... /lectures/ --tags=cardiology,exam2024")
        print("\n  # Custom styling")
        print("  python script.py sk-abc... lecture.pdf --style=background=#1a1a1a,text_color=#ffffff,cloze_color=#00ff00")
        return
    
    api_key = sys.argv[1]
    path = sys.argv[2]
    
    # Parse options
    resume = "--no-resume" not in sys.argv
    single_card_mode = "--multiple-cloze" not in sys.argv  # Inverted logic
    budget_mode = "--budget" in sys.argv
    test_mode = "--test-mode" in sys.argv
    add_hints = "--no-hints" not in sys.argv  # Inverted logic
    
    # Parse compression level
    compression_level = "high"  # Default
    for arg in sys.argv:
        if arg.startswith("--compress="):
            level = arg.split("=", 1)[1]
            if level in ["none", "low", "medium", "high"]:
                compression_level = level
            else:
                print(f"⚠️ Invalid compression level '{level}', using 'high'")
    
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
    
    # Show mode information
    if budget_mode:
        print(f"\n💰 Budget mode: Skipping refinement stage")
    else:
        print(f"\n🧠 Advanced mode: AI critique and refinement enabled")
    
    if test_mode:
        print("🧪 Test mode enabled - will pause before each API call")
    
    generator = MedicalAnkiGenerator(
        api_key, 
        single_card_mode=single_card_mode,
        custom_tags=custom_tags,
        card_style=card_style,
        compression_level=compression_level,
        test_mode=test_mode,
        add_hints=add_hints
    )
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        generator.process_lecture(path, resume=resume, budget_mode=budget_mode)
    elif os.path.isdir(path):
        generator.process_folder(path, resume=resume, budget_mode=budget_mode)
    else:
        print("❌ Please provide a valid PDF file or folder path")


if __name__ == "__main__":
    main()