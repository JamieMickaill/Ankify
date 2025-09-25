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
5. NEVER create cloze deletions where the answer is given elsewhere in the card (e.g., in parentheses, as an abbreviation expansion)
6. When abbreviations are important, test the abbreviation OR the full name, not both
7. Always include disease/condition context when listing causes, symptoms, or treatments
8. Check for formatting errors in cloze syntax (no spaces between colons)
9. Don't test irrelevant details from other conditions mentioned in passing

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

❌ BAD: "Routine coagulation monitoring with {{c1::INR}} is {{c2::not required}} for dabigatran, rivaroxaban or apixaban but is mandatory for warfarin"
→ Problem: "but is mandatory" reveals that c2 is "not required"
✅ BETTER: "{{c1::INR}} monitoring is {{c2::not required}} for DOACs, {{c2::required}} for warfarin"

❌ BAD: "Life-threatening bleeding on rivaroxaban can be treated with {{c1::andexanet alfa}} (or {{c1::PCC}} if andexanet is unavailable)"
→ Problem: "if andexanet is unavailable" gives away first cloze
✅ BETTER: "Life-threatening bleeding on rivaroxaban: use {{c1::andexanet alfa}} (first-line) or {{c1::PCC}} (if first-line unavailable)"

❌ BAD: "Haemostasis is the {{c1::physiological}} sealing of vessel defects"
→ Problem: Testing that something is "physiological" is not meaningful
✅ BETTER: "Haemostasis involves {{c1::platelet adhesion}} and {{c1::fibrin deposition}}"

❌ BAD: "Karyotyping detects {{c1::t(14;18) in follicular lymphoma::translocation}}"
→ Problem: Both translocation AND disease in one cloze is too ambiguous
✅ BETTER: "Follicular lymphoma shows {{c1::t(14;18)}} translocation" OR "{{c1::t(14;18)}} is seen in {{c2::follicular lymphoma}}"

❌ BAD: "Successful outcomes depend on initiating interventions at the {{c1::right time}} — {{c1::neither too early nor too late}}"
→ Problem: Not a testable medical concept
✅ SKIP THIS CONTENT - not assessable

❌ BAD: "Psychotic disorders have a prevalence of {{c1::a few percent}} (≈3–5%)"
→ Problem: Answer given in parentheses defeats the learning purpose
✅ BETTER: "Psychotic disorders have a lifetime prevalence of {{c1::3-5%}} of the population"

❌ BAD: "On computed tomography ({{c1::CT}})/magnetic resonance imaging ({{c1::MRI}}), a subdural..."
→ Problem: Full names given before abbreviations defeats the cloze
✅ BETTER: "On {{c1::CT}} or {{c1::MRI}}, a subdural haematoma is {{c2::crescent-shaped}}"

❌ BAD: "In patients >60 years, common causes include {{c1::atrial fibrillation}}"
→ Problem: Missing context - what disease are these causes of?
✅ BETTER: "In patients >60 years, common causes of {{c2::stroke}} include {{c1::atrial fibrillation}}"

❌ BAD: "The {{c1::Fisher}} grade correlates with risk of {{c2::cerebral vasospasm}}"
→ Problem: Missing SAH context makes card ambiguous
✅ BETTER: "In {{c3::subarachnoid haemorrhage}}, the {{c1::Fisher}} grade predicts {{c2::vasospasm}} risk"

❌ BAD: "For {{c1::minor stroke}}, start {{c2::aspirin}} plus {{c2: :clopidogrel}}"
→ Problem: Formatting error with space between colons
✅ BETTER: "For {{c1::minor stroke}}, start {{c2::aspirin}} plus {{c2::clopidogrel}}"


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

CONTEXT PRESERVATION RULES:
- When testing causes/symptoms/treatments, ALWAYS specify what condition they relate to
- Bad: "Common causes include {{c1::atrial fibrillation}}"
- Good: "Common causes of stroke include {{c1::atrial fibrillation}}"
- When testing grading systems or scores, include what condition they assess
- Bad: "The {{c1::Fisher}} grade predicts vasospasm"
- Good: "In SAH, the {{c1::Fisher}} grade predicts {{c2::vasospasm}}"

ANSWER LEAKAGE PREVENTION:
- Never include the answer in parentheses after a cloze
- Never spell out abbreviations before testing them
- Remove redundant information that gives away cloze answers
- If showing ranges or examples, incorporate them INTO the cloze or remove them

CRITICAL REMINDERS:
- Test ALL important facts in a statement, not just one
- Ensure abbreviations are spelled out at least once
- Provide sufficient context for self-contained understanding
- Focus on clinically relevant information, not research priorities
- Avoid cards where remaining context reveals the cloze deletion

"""
    
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

1) For EACH slide, extract MEDICAL FACTS AND CONCEPTS that medical students need to know for exams and clinical practice (not meta-information about the lecture).

2) FOCUS on creating cards for:
- Core pathophysiology and disease mechanisms
- Key clinical features and presentations
- First-line treatments and management principles
- Important differential diagnoses
- High-yield diagnostic approaches
- Clinical decision-making concepts
- Best practice guidelines (not minute details)
- Important contraindications and safety considerations

3) AVOID creating cards for:
- Specific radiation doses or technical parameters
- Names/authors of individual studies (unless landmark trials)
- Overly specialized procedural details
- Research methodology minutiae
- Historical facts unless clinically relevant
- Subspecialty-specific technical details
- Future research priorities or ongoing trials

4) SKIP slides that only contain:
- Title/topic announcements
- Learning objectives/outcomes lists
- Lecture outlines or agendas
- Speaker introductions
- References/bibliography

5) Emphasize:
- "Why" and "when" rather than exact numbers
- Clinical reasoning and decision pathways
- Comparative effectiveness
- Practical clinical applications

6) ENSURE each card:
- Can be understood without seeing the original lecture
- Has specific, unambiguous cloze deletions
- Tests one clear concept per cloze
- Provides enough context to identify the answer
- Tests ALL key facts (definitions, mechanisms, effects, durations)
- Uses same cloze number (c1) for related binary/mutually exclusive choices
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
   - Multiple symptoms/signs of the same condition/disease
   - Multiple examples of the same category
   - Related anatomical structures in a list
   - Multiple drugs with the same indication
   - Related lab values or ranges
   - Components of a syndrome or criteria
   - Chromosomes in translocations (e.g., both 8 and 14 in t(8;14))
   - Markers that are co-expressed (e.g., CD5 and CD23)
   - Binary/mutually exclusive choices (prophylaxis vs treatment)

2. SEPARATE INTO DIFFERENT CLOZE NUMBERS:
   - Test name vs test result
   - Drug name vs mechanism of action
   - Drug name vs related indication 
   - Drug name vs its antidote/reversal agent
   - Disease name vs symptoms/signs/tissue/location 
   - Disease name vs treatment
   - Pathology/disease vs its mechanism
   - Anatomical location vs pathology type 
   - Anatomical structure/vessel vs resulting syndrome
   - Cause vs effect
   - Structure vs function
   - Normal vs abnormal values
   - Factor levels vs severity grades
   - Variant types vs prognosis
   - Management action vs its complication/outcome
   - Target value vs drug used to achieve it
   - Grading system name vs what it predicts

3. EXAMPLES OF GOOD FINE-TUNED GROUPING:

✅ GOOD: "Symptoms of DKA include {{c1::polyuria}}, {{c1::polydipsia}}, and {{c1::weight loss}}, with labs showing {{c2::glucose >250}} and {{c2::pH <7.3}}"
→ Groups symptoms together (c1) and lab values together (c2)

✅ GOOD: "{{c1::Metformin}} and {{c1::sulfonylureas}} are oral agents, while {{c2::insulin}} is injectable for diabetes"
→ Groups oral agents together, separates by route

✅ GOOD: "The {{c1::indirect antiglobulin test (IAT)}} detects {{c2::unexpected red-cell antibodies}}"
→ Separates test from what it detects to avoid ambiguity

✅ GOOD: "{{c2::Aspirin}} inhibits {{c1::thromboxane synthesis}}, while {{c2::clopidogrel}} blocks {{c1::ADP receptor}}"
→ Groups drugs together and mechanisms together

✅ GOOD: "Burkitt lymphoma involves chromosome {{c1::8}} (c-MYC) and {{c1::14}} (IgH)"
→ Groups related chromosomes in translocation

✅ GOOD: "Co-expression of {{c1::CD5}} and {{c1::CD23}} characterizes {{c2::CLL/SLL}}"
→ Groups co-expressed markers together

✅ GOOD: "In Hodgkin lymphoma, {{c1::lymphocyte-rich}} has {{c2::best}} prognosis, {{c1::lymphocyte-depleted}} has {{c2::worst}} prognosis"
→ Groups variants together, groups prognoses together

✅ GOOD: "Factor levels {{c1::<1%}} indicate {{c2::severe}} haemophilia, {{c1::1-5%}} indicate {{c2::moderate}}, {{c1::5-40%}} indicate {{c2::mild}}"
→ Groups factor levels together, groups severities together

✅ GOOD: Risk factors for eating disorders include {{c1::female or trans male gender}}, {{c1::puberty/adolescence}}, {{c1::perfectionism or autism spectrum traits}}, participation in {{c1::weight‑ or aesthetics‑focused sports (ballet, gymnastics, figure skating)}}, {{c1::trauma or major life events}}, {{c1::family history of eating disorders}}, and a {{c1::highly competitive environment}}.
→ Groups risk factors together

4. EXAMPLES TO AVOID:

❌ BAD: "The antibody screen looks for {{c1::unexpected antibodies}} using {{c1::IAT}}"
→ Too ambiguous when both are hidden - separate test from target

❌ BAD: "DKA presents with {{c1::polyuria}}, {{c2::polydipsia}}, {{c3::weight loss}}"
→ Unnecessary separation of related symptoms

❌ BAD: "Chromosome {{c1::8}} and {{c2::14}} in Burkitt lymphoma"
→ Should group chromosomes in same translocation

❌ BAD: Psychedelics/related agents—{{c1::ketamine::drug/medication}}, {{c2::MDMA::drug/medication}}, {{c3::psilocybin::drug/medication}}, and {{c4::cannabidiol::drug/medication}}—are {{c5::experimental::evidence status}} for eating disorders and should be considered only {{c6::within clinical trials::use setting}} due to {{c7::safety and legal considerations}}.
→ Drug names for the same indication must be grouped together

❌ BAD: Medical complications of {{c1::anorexia nervosa}} include {{c2::bradycardia and hypotension::clinical feature}}, {{c3::arrhythmias::clinical feature}}, {{c4::amenorrhoea/infertility::clinical feature}}, {{c5::osteopenia/osteoporosis::clinical feature}}, {{c6::anaemia/leukopenia::clinical feature}}, {{c7::renal impairment/dehydration::clinical feature}}, and {{c8::constipation::clinical feature}}.
→ Symptoms for the same disease must be grouped together

❌ BAD: In {{c1::bulimia nervosa}}, purging can cause {{c2::dental enamel erosion::complication}}, {{c3::parotid enlargement::complication}}, {{c4::hypokalaemic metabolic alkalosis::complication}}, {{c5::oesophagitis/Mallory–Weiss tears::complication}}, and {{c6::renal complications::complication}}.
→ Complications for the same disease must be grouped together

❌ BAD: {{c1::Binge‑eating disorder}} is strongly associated with {{c2::obesity}}, {{c3::metabolic syndrome}}, and {{c4::type 2 diabetes mellitus}}.
→ Associations for the same disease must be grouped together

❌ BAD: Physical harm minimisation in eating disorders includes routine monitoring of {{c1::vital signs and blood glucose::vitals/lab}}, {{c2::electrolytes (especially phosphate, potassium, magnesium)::investigation}}, {{c3::ECG::investigation}}, {{c4::bowel function::clinical monitoring}}, and {{c5::bone and dental health::screening}}.
→ Harm minimisation strategies for the same disease must be grouped together

❌ BAD: Refeeding syndrome is marked by {{c1::hypophosphataemia::biochemical hallmark}} with risk of {{c2::arrhythmias::severe}} and {{c3::heart failure::severe}}; prevent by {{c4::slow refeeding::management}}, {{c5::close electrolyte monitoring/repletion::management}} and {{c6::thiamine supplementation::management}}.
→ Risks should be grouped together (arrythmia/heart failure) and prevention strategies should be grouped together (slow refeed, electrolyte monitoring)

❌ BAD: "Subcutaneous heparin for {{c1::prophylaxis}}, IV for {{c2::treatment}}"
→ Binary choice makes other obvious - use same cloze

❌ BAD: "CD5{{c1::+}} and CD23{{c2::+}} in CLL"
→ Co-expressed markers should be grouped

❌ BAD: "{{c1::Quetiapine::drug/medication}} is effective for both {{c1::acute mania::indication}} and {{c1::bipolar depression::indication}}, and can be used for {{c1::maintenance::indication}}"
→ Drug and indication must ALWAYS be separate cloze numbers

❌ BAD: "{{c1::Lamotrigine::drug/medication}} is useful for {{c1::bipolar depression::indication}} and {{c1::maintenance::indication}}, but is {{c1::not effective}} for {{c1::acute mania::indication}}"
→ Drug, indications, and efficacy status should be separate

❌ BAD: "{{c1::Lithium::drug/medication}} treats {{c1::acute mania::indication}} and is effective for {{c1::maintenance::indication}} in bipolar disorder"
→ Drug name and indications must be separate

❌ BAD: "The {{c1::limbus::location}} is the most common site for {{c1::ocular surface neoplasia::tumor type}}"
→ Anatomical location and pathology must ALWAYS be separate

❌ BAD: "Reduced corneal innervation can lead to {{c1::dry eye (reduced tear secretion)::clinical feature}}, {{c1::epithelial ulceration::clinical feature}}, and {{c1::scarring::sequela}}—features of {{c1::neurotrophic keratopathy::diagnosis}}"
→ Diagnosis/disease name must be separate from its symptoms/signs

❌ BAD: "{{c1::Herpes simplex virus::pathogen}} remains latent in the {{c1::trigeminal ganglion::site of latency}} and can reactivate and travel along nerves to infect the {{c1::cornea::target tissue}}, leading to {{c1::herpetic keratitis::disease}} and scarring"
→ Pathogen, anatomical sites, and resulting disease must all be separate

❌ BAD: Life events involving {{c1::loss}} or {{c1::humiliation/threat}} show a strong association with the {{c3::onset}} of major depression.
→ Grouping criteria is good, but cloze does not have c2, while c3 is included

❌ BAD: "A {{c1::posterior inferior cerebellar artery (PICA)}} infarct causes {{c1::lateral medullary (Wallenberg) syndrome}}"
→ Vessel and syndrome must be separate - too ambiguous when both hidden

❌ BAD: "{{c1::Choroid plexus tumours}} can cause hydrocephalus by {{c1::overproduction of CSF}}"
→ Tumor type and mechanism must be separate

❌ BAD: "Reverse {{c1::warfarin}} with {{c1::prothrombin complex concentrate}} plus {{c1::vitamin K}}"
→ Drug and its antidotes must be separate cloze numbers

❌ BAD: "Secondary prevention aims for {{c1::LDL <1.8 mmol/L}} using {{c1::high-intensity statins}}"
→ Target value and treatment must be separate

❌ BAD: "The {{c1::Fisher}} grade is a {{c1::CT-based}} score that correlates with risk of {{c2::cerebral vasospasm}}"
→ Grade name and imaging modality should be separate, plus missing SAH context

5. DECISION FRAMEWORK:
   - Can the blanks be logically filled when hidden together? → Use same cloze
   - Would hiding them together make the card unanswerable? → Use different cloze
   - Are they examples of the same category? → Use same cloze
   - Do they represent cause and effect? → Use different cloze
   - Are they binary/mutually exclusive choices? → Use same cloze
   - Are they components of same translocation/co-expression? → Use same cloze
   - Is one a drug/medication and the other an indication? → ALWAYS use different cloze
   - Is one a location and the other a pathology? → ALWAYS use different cloze
   - Is one a disease/pathogen and the other symptoms/location? → ALWAYS use different cloze

6. ERR ON THE SIDE OF GROUPING:
   When uncertain (except for the mandatory separations above), prefer fewer cloze numbers. It's easier to manually create additional cards later than to merge multiple cards.

REMEMBER: The goal is intelligent grouping that creates fewer cards while maintaining the ability to answer each card without excessive ambiguity. However, certain relationships (drug-indication, location-pathology, disease-symptoms) must ALWAYS be kept separate to maintain clinical learning effectiveness."""

    @staticmethod
    def get_json_format() -> str:
        """Get the expected JSON format for responses."""
        return """

Format your response as a JSON array of flashcard objects, where each object has:
- "text": The complete text with cloze deletions in {{c1::answer}} format (can have multiple clozes {{c1::}}, {{c2::}}, etc.)
- "facts": Array of the key facts being tested
- "context": Brief context about why this is important FOR A MEDICAL STUDENT
- "clinical_relevance": Optional field for clinical pearls or practical applications

Return a JSON array with one object per slide:
Example:
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

IMPORTANT: Include ALL slides / page_num in your response, even if a slide has no relevant medical content (return empty cards array for that slide).
Make cards self-contained with clear, unambiguous cloze deletions that can be answered without lecture context.

"""

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
- "vitals" - for vital signs (temperature, heart rate, blood pressure)
- "symptom" - for subjective patient complaints
- "severe" - for severe complications or conditions
- "management" - for treatment actions

EXAMPLES:
✅ "Tamoxifen is preferred for {{c1::premenopausal::hormonal status}} women"
Why: Initially broad category of types of women is narrowed down to those of different hormonal status. Does not give away answer.
✅ "TNBC shows {{c1::ER < 1%, PR < 1%::receptor expression}}"
Why: TNBC could show anything (initially ambiguous), hint narrowed down to those of different receptor expression. Does not give away answer.
✅ "Adjuvant therapy is given for {{c1::5 years::duration}} (10 if high-risk)"
Why: Initial interpretation could be indication, demographic, pathological subtype -> hint narrows down to duration without giving away answer.
✅ "{{c1::CDK4/6 inhibitors::drug class}} like palbociclib block cell cycle"
✅ "During transfusion, sudden onset of {{c1::fever or hypothermia::vitals}}, {{c1::rigors::symptom}}, {{c1::tachycardia::vitals}}, or progression to {{c1::shock/multi-organ failure::severe}} should raise suspicion"
Why: More specific hints (vitals vs symptom vs severe) reduce ambiguity without revealing answers

AVOID:
❌ Valueless hints: The prostate cancer survivorship essentials framework lists six domains: {{c1::Health promotion &amp; advocacy::domain}}, {{c1::Evidence‑based interventions::domain}}, {{c1::Personal agency::domain}}, {{c1::Vigilance::domain}}, {{c1::Care coordination::domain}}, and {{c1::Shared management::domain}}.
WHY: Domain does not give any hint at all, "domain" is a generic placeholder for anything and is included in the cloze already. NO REDUCTION IN AMBIGUITY

❌ Generic placeholders: Priority actions for men with prostate cancer include enhancing {{c1::patient‑clinician communication::action}}, developing a {{c1::survivorship toolkit::action}}, expanding {{c1::multi‑modal care::action}}, reducing {{c1::out‑of‑pocket costs::action}}, promoting {{c1::exercise::action}}, harnessing {{c1::technology for access::action}}, and building {{c1::specialist outreach capacity::action}}.
WHY: "action" is a generic placeholder that provides no hint value and is already mentioned in the cloze

❌ Inaccurate hints: Whole blood is roughly {{c1::55 % plasma::percentage}} and {{c1::45 % formed elements (red cells, white cells, platelets)::percentage}}.
WHY: The deletion is not just a percentage but a percentage AND a component of blood. Hints must accurately describe the WHOLE cloze deletion content

❌ Redundant hints: {{c1::nausea::nausea symptom}}

❌ Hints that give away the answer: In HITTS, platelet transfusion is {{c1::contra-indicated::contraindication}}
WHY: The hint "contraindication" is literally the same as the cloze "contra-indicated" - provides no value

❌ Hints placed outside cloze: Key features include {{c1::shortness of breath}}, {{c1::pleuritic chest pain}}::clinical feature
WHY: Hint must be inside EACH cloze deletion: {{c1::shortness of breath::clinical feature}}, {{c1::pleuritic chest pain::clinical feature}}

OVERARCHING GOAL: Specific hints without giving away the answer are optimal. This is challenging however, and you should err on the side of ambiguity rather than specificity."""

class MedicalAnkiGenerator:
    def __init__(self, openai_api_key: str, single_card_mode: bool = True,  # Changed default to True
                custom_tags: Optional[List[str]] = None, card_style: Optional[Dict] = None,
                compression_level: str = "high",  # Changed default to "high"
                test_mode: bool = False, 
                add_hints: bool = True, # Changed default to True, removed batch_mode and preserve_quality
                flex_mode: bool = False):  
        self.api_key = openai_api_key
        self.single_card_mode = single_card_mode
        self.custom_tags = custom_tags or []
        self.card_style = card_style or {}
        self.compression_level = compression_level
        self.test_mode = test_mode
        self.flex_mode = flex_mode
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
    
    
    def _build_batch_analysis_prompt(self, num_slides: int, lecture_name: str) -> str:
        """Build the batch analysis prompt using centralized templates."""
        cloze_instruction = PromptTemplates.get_cloze_instruction(self.single_card_mode)
        
        return f"""You are analyzing {num_slides} slides from a medical lecture on "{lecture_name}" to create Anki flashcards specifically for MEDICAL STUDENTS preparing for exams and clinical practice.
        
    Your PRIMARY GOAL: Create high-quality, unambiguous Anki flashcards that help medical students retain essential knowledge for both exams and patient care.

{PromptTemplates.get_common_rules()}

{PromptTemplates.get_cloze_examples()}

{PromptTemplates.get_advanced_principles()}

{PromptTemplates.get_percentage_guidelines()}

{PromptTemplates.get_content_focus()}

{cloze_instruction}

{PromptTemplates.get_json_format()}
"""

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
        service_tier = "flex" if self.flex_mode else "default"

        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 100000,
            "reasoning_effort": "high",
            "service_tier": f"{service_tier}",
        }
        
        self.logger.info(f"Sending batch request with {len(content)} content items")

        
        
        # Dynamic timeout based on number of slides (30 seconds per slide + 300 second buffer)
        timeout_seconds = max(600, (len(images) * 30) + 300)
        if self.flex_mode: timeout_seconds*=2

        print(f"⏱️ Timeout set to {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes) for {len(images)} slides")

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
                    timeout=timeout_seconds
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
        """Use AI model to critique and refine cards in three separate stages."""
        print("\n🔬 Starting multi-stage optimization process...")
        self.logger.info(f"Starting multi-stage optimization for lecture: {lecture_name}")
        
        # Test mode check
        if self.test_mode:
            print("\n🔍 Ready to start multi-stage optimization")
            print("Press Enter to continue (or 'skip' to skip all optimization, 'quit' to exit)...")
            user_input = input().strip().lower()
            if user_input == 'skip':
                print("⏭️ Skipping all optimization")
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

        all_decisions = []
        hint_decisions = []
        grouping_decisions = []
        ambiguity_decisions = []

        # Stage 1: Refinement only
        print("\n📝 Stage 1/4: Initial refinement and quality improvement...")
        prompt = self._build_critique_prompt_refinement_only(lecture_name, cards_for_review)

        service_tier = "flex" if self.flex_mode else "default"
        
        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 100000,
            "reasoning_effort": "high",
            "service_tier": f"{service_tier}",
        }
        
        max_retries = 3
        refined_cards = None
        decisions = []
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 30 * attempt
                    print(f"\n⏳ Retry {attempt}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=1200
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        result = json.loads(json_match.group())
                        refined_cards = result.get('refined_cards', [])
                        stage1_decisions = result.get('decisions', [])
                        for decision in stage1_decisions:
                            decision['stage'] = 'refinement'
                        all_decisions.extend(stage1_decisions)
                        break
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
        
        if not refined_cards:
            print("⚠️ Stage 1 failed, using original cards")
            self._save_refinement_logs(refinement_log_file, lecture_name, total_original_cards, refined_cards, all_decisions,hint_decisions,grouping_decisions,ambiguity_decisions)
            return all_cards_data
        
        print(f"✅ Stage 1 complete: {total_original_cards} → {len(refined_cards)} cards")
        
        # Stage 2: Add hints (if enabled)
        if self.add_hints and refined_cards:
            print("\n💡 Stage 2/4: Adding descriptive hints...")
            prompt = self._build_hints_only_prompt(lecture_name, refined_cards)
            
            payload = {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 100000,
                "reasoning_effort": "high",
                "service_tier": f"{service_tier}",
            }
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = 30 * attempt
                        print(f"\n⏳ Retry {attempt}/{max_retries} after {wait_time}s wait...")
                        time.sleep(wait_time)
                    
                    response = self.session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=1200
                    )
                    
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content']
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            result = json.loads(json_match.group())
                            cards_with_hints = result.get('cards_with_hints', [])
                            hint_decisions = result.get('hint_decisions', [])
                            if cards_with_hints:
                                refined_cards = cards_with_hints
                                for decision in hint_decisions:
                                    decision['stage'] = 'hints'
                                all_decisions.extend(hint_decisions)
                                print(f"✅ Stage 2 complete: Hints added to {len(refined_cards)} cards")
                            break
                            
                except Exception as e:
                    print(f"\n❗ Error during hint addition: {str(e)}")
                    if attempt == max_retries - 1:
                        print("⚠️ Continuing without hints")
                        self._save_refinement_logs(refinement_log_file, lecture_name, total_original_cards, refined_cards, all_decisions,hint_decisions,grouping_decisions,ambiguity_decisions)

        else:
            print("\n⏭️ Stage 2/4: Skipping hints (disabled)")
        
        # Stage 3: Optimize grouping (if not in single card mode)
        if not self.single_card_mode and refined_cards:
            print("\n🔄 Stage 3/4: Optimizing cloze grouping...")
            prompt = self._build_grouping_only_prompt(lecture_name, refined_cards)
            
            payload = {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 100000,
                "reasoning_effort": "high",
                "service_tier": f"{service_tier}",
            }
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = 30 * attempt
                        print(f"\n⏳ Retry {attempt}/{max_retries} after {wait_time}s wait...")
                        time.sleep(wait_time)
                    
                    response = self.session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=1200
                    )
                    
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content']
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            result = json.loads(json_match.group())
                            optimized_cards = result.get('optimized_cards', [])
                            grouping_decisions = result.get('grouping_decisions', [])
                        
                            if optimized_cards:
                                refined_cards = optimized_cards
                                for decision in grouping_decisions:
                                    decision['stage'] = 'grouping'
                                all_decisions.extend(grouping_decisions)
                                print(f"✅ Stage 3 complete: Grouping optimized for {len(refined_cards)} cards")
                            break
                            
                except Exception as e:
                    print(f"\n❗ Error during grouping optimization: {str(e)}")
                    if attempt == max_retries - 1:
                        print("⚠️ Using cards without grouping optimization")
                        self._save_refinement_logs(refinement_log_file, lecture_name, total_original_cards, refined_cards, all_decisions,hint_decisions,grouping_decisions,ambiguity_decisions)
        else:
            print("\n⏭️ Stage 3/4: Skipping grouping (single card mode)")
        

        # Stage 4: Final check
        if refined_cards:
            print("\n🔍 Stage 4/4: Final ambiguity check...")
            prompt = self._build_ambiguity_check_prompt(lecture_name, refined_cards)
            
            payload = {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 100000,
                "reasoning_effort": "high",
                "service_tier": f"{service_tier}",
            }
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = 30 * attempt
                        print(f"\n⏳ Retry {attempt}/{max_retries} after {wait_time}s wait...")
                        time.sleep(wait_time)
                    
                    response = self.session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=1200
                    )
                    
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content']
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            result = json.loads(json_match.group())
                            checked_cards = result.get('checked_cards', [])
                            ambiguity_decisions = result.get('ambiguity_decisions', [])
                
                            if checked_cards:
                                refined_cards = checked_cards
                                # Add ambiguity decisions to the decisions log
                                for decision in ambiguity_decisions:
                                    if decision.get('action') == 'modified':
                                        decisions.append({
                                            'action': 'modified',
                                            'stage': 'ambiguity_check',
                                            'original_text': decision.get('original_text'),
                                            'new_text': decision.get('new_text'),
                                            'reason': decision.get('reason')
                                        })
                                print(f"✅ Stage 4 complete: {len([d for d in ambiguity_decisions if d.get('action') == 'modified'])} cards modified for clarity")
                            break
                            
                except Exception as e:
                    print(f"\n❗ Error during ambiguity check: {str(e)}")
                    if attempt == max_retries - 1:
                        print("⚠️ Continuing without ambiguity check")
        else:
            print("\n⏭️ Stage 4/4: Skipping ambiguity check (no cards)")
        # Save and process refinement results
        self._save_refinement_logs(
        refinement_log_file, 
        lecture_name, 
        total_original_cards, 
        refined_cards, 
        all_decisions,
        hint_decisions,
        grouping_decisions,
        ambiguity_decisions
    )
        
        # Validate and organize refined cards
        return self._process_refined_cards(refined_cards)
     
    def _build_critique_prompt_refinement_only(self, lecture_name: str, cards_for_review: List[Dict]) -> str:
        """Build the critique prompt without hints or grouping instructions."""
        
        return f"""You are an expert medical educator reviewing cloze deletion flashcards from a lecture on "{lecture_name}".

    CRITICAL INSTRUCTIONS:
    1. ALL cards MUST remain in cloze deletion format 
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
        "original_index": [0]
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
        "original_index": [5, 8],
        "original_texts": ["Card about CDK4/6 inhibitors", "Another card about CDK4/6 inhibitors"],
        "reason": "Both cards tested the same concept about CDK4/6 inhibitors"
        }}
    ]
    }}

    ⚠️ REMEMBER: The goal is to REFINE cards while MAINTAINING their good cloze deletion patterns. Do not make cards more ambiguous in the name of brevity. Each card must be answerable without having seen the lecture."""

    def _build_hints_only_prompt(self, lecture_name: str, cards: List[Dict]) -> str:
        """Build prompt for adding hints only."""
        return f"""You are adding descriptive hints to medical flashcards from "{lecture_name}" for MEDICAL STUDENTS.

    PRIMARY GOAL: Ensure every Anki flashcard has appropriate hints to help medical students learn effectively.

    TASK: Add hints to ALL cloze deletions where they would reduce ambiguity. A hint should be present for EVERY cloze unless it would provide no value.

    {PromptTemplates.get_hint_instructions()}

    CRITICAL RULES:
    - DO NOT change the card text except for adding hints
    - DO NOT change cloze numbers
    - MUST add hints to ALL clozes unless they provide zero value
    - Provide a reason for each hint decision
    - Format: {{{{c1::answer::hint}}}}

    Current cards WITHOUT hints:
    {json.dumps(cards, indent=2)}

    Return JSON:
    {{
        "cards_with_hints": [
            {{
                "slide": 1,
                "text": "Card text with {{{{c1::answer::appropriate_hint}}}} added",
                "facts": ["fact1"],
                "context": "Context",
                "clinical_relevance": "Clinical pearl"
            }}
        ],
        "hint_decisions": [
            {{
                "card_index": 0,
                "cloze": "c1::answer",
                "hint_added": "appropriate_hint",
                "reason": "Added 'drug class' hint to distinguish from other treatment options"
            }},
            {{
                "card_index": 0,
                "cloze": "c2::obvious_answer",
                "hint_added": null,
                "reason": "No hint needed - answer is unambiguous in context"
            }}
        ]
    }}"""

    def _build_grouping_only_prompt(self, lecture_name: str, cards: List[Dict]) -> str:
        """Build prompt for optimizing cloze grouping only."""

        return f"""You are optimizing cloze number grouping for medical flashcards from "{lecture_name}" specifically for MEDICAL STUDENTS.

    PRIMARY GOAL: Create Anki flashcards that help medical students effectively learn and retain clinical knowledge.

    TASK: Optimize cloze number assignments to create the fewest cards while maintaining clarity.

    {PromptTemplates.get_fine_tuned_cloze_principles()}

    CRITICAL RULES:
    - DO NOT change card text or remove hints
    - ONLY adjust cloze numbers (c1, c2, c3, etc.)
    - Provide a reason for each grouping decision

    Current cards WITH hints:
    {json.dumps(cards, indent=2)}

    Return JSON:
    {{
        "optimized_cards": [
            {{
                "slide": 1,
                "text": "Card with optimized {{{{c1::answer::hint}}}} numbers",
                "facts": ["fact1"],
                "context": "Context",
                "clinical_relevance": "Clinical pearl",
                "grouping_reason": "Grouped symptoms together as they represent related clinical features of the same condition"
            }}
        ],
        "grouping_decisions": [
            {{
                "card_index": 0,
                "original_cloze_pattern": "c1, c2, c3 for individual symptoms",
                "new_cloze_pattern": "all c1 for grouped symptoms",
                "reason": "Symptoms of the same disease should be learned together"
            }}
        ]
    }}"""

    def _build_ambiguity_check_prompt(self, lecture_name: str, cards: List[Dict]) -> str:
        """Build prompt for final ambiguity check."""
        return f"""You are performing a final ambiguity check on medical flashcards from "{lecture_name}" for MEDICAL STUDENTS.

    PRIMARY GOAL: Ensure every Anki flashcard is unambiguous and appropriately challenging for medical student learning.

    TASK: For EACH cloze card, perform this systematic check:

    1. Write out the card WITHOUT cloze deletions (replace with * or _)
    2. Check if context/topic is clear (ambiguity = >100 possible answers for a medical student)
    3. Check if deletions are guessable within a narrow domain
    4. Check if deletions are too obvious (mutually exclusive options not deleted, acronyms spelled out)
    5. Check if hints sufficiently reduce ambiguity
    6. Check if hints are too obvious
    7. Take action: modify text/hints OR mark for removal if obvious

    EXAMPLE ANALYSIS:
    Original: A careful history guides {{{{c1::seizure classification}}}}, {{{{c1::epilepsy syndrome diagnosis}}}}, and {{{{c1::antiseizure medication titration::management}}}}.
    Without cloze: A careful history guides *, *, and _.
    Context check: FAIL - seizure context missing, >100 possible clinical scenarios
    Guessability: FAIL - too broad without disease context
    Obviousness: PASS - not obvious
    Hints check: FAIL - "management" too vague without disease context
    Hints Obviousness: PASS - not obvious
    Action: ADD CONTEXT
    New: A careful seizure history guides {{{{c1::seizure classification::seizure diagnosis}}}}, {{{{c1::epilepsy syndrome diagnosis::epilepsy diagnosis}}}}, and {{{{c1::antiseizure medication titration::management}}}}.

    Current cards:
    {json.dumps(cards, indent=2)}

    Return JSON:
    {{
        "checked_cards": [
            {{
                "slide": 1,
                "text": "Final checked card text",
                "facts": ["fact1"],
                "context": "Context",
                "clinical_relevance": "Clinical pearl"
            }}
        ],
        "ambiguity_decisions": [
            {{
                "card_index": 0,
                "original_text": "Original card text",
                "without_cloze": "Card with * and _ replacing clozes",
                "context_clear": false,
                "guessability": "too broad",
                "obviousness": "appropriate",
                "hints_sufficient": false,
                "hints_obvious": false,
                "action": "modified",
                "new_text": "Modified card with better context",
                "reason": "Added disease context to reduce ambiguity from >100 to <10 possible answers"
            }}
        ]
    }}"""

    def _save_refinement_logs(self, refinement_log_file: Path, lecture_name: str, 
                                        total_original_cards: int, refined_cards: List[Dict], 
                                        all_decisions: List[Dict], hint_decisions: List[Dict],
                                        grouping_decisions: List[Dict], ambiguity_decisions: List[Dict]):
        """Save comprehensive refinement logs from all stages."""
        refinement_data = {
            "lecture": lecture_name,
            "timestamp": datetime.now().isoformat(),
            "original_count": total_original_cards,
            "refined_count": len(refined_cards),
            "all_decisions": all_decisions,
            "stage_breakdown": {
                "stage1_refinement": {
                    "decisions": [d for d in all_decisions if d.get('stage') == 'refinement'],
                    "removed": len([d for d in all_decisions if d.get('stage') == 'refinement' and d.get('action') == 'removed']),
                    "merged": len([d for d in all_decisions if d.get('stage') == 'refinement' and d.get('action') == 'merged']),
                    "modified": len([d for d in all_decisions if d.get('stage') == 'refinement' and d.get('action') == 'modified'])
                },
                "stage2_hints": {
                    "decisions": hint_decisions,
                    "hints_added": len([d for d in hint_decisions if d.get('hint_added')])
                },
                "stage3_grouping": {
                    "decisions": grouping_decisions,
                    "regrouped": len(grouping_decisions)
                },
                "stage4_ambiguity": {
                    "decisions": ambiguity_decisions,
                    "modified": len([d for d in ambiguity_decisions if d.get('action') == 'modified'])
                }
            },
            "summary": {
                "total_decisions": len(all_decisions),
                "removed": len([d for d in all_decisions if d.get('action') == 'removed']),
                "merged": len([d for d in all_decisions if d.get('action') == 'merged']),
                "modified": len([d for d in all_decisions if d.get('action') == 'modified']),
                "hints_added": len([d for d in hint_decisions if d.get('hint_added')]),
                "grouping_changes": len(grouping_decisions),
                "ambiguity_fixes": len([d for d in ambiguity_decisions if d.get('action') == 'modified'])
            }
        }
        
        # Save JSON log
        with open(refinement_log_file, 'w', encoding='utf-8') as f:
            json.dump(refinement_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Comprehensive refinement log saved to: {refinement_log_file}")
        
        # Create human-readable summary
        summary_file = refinement_log_file.with_suffix('.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Multi-Stage Refinement Summary for {lecture_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original cards: {total_original_cards}\n")
            f.write(f"Final refined cards: {len(refined_cards)}\n")
            if total_original_cards > 0:
                reduction_percent = (1 - len(refined_cards)/total_original_cards) * 100
                f.write(f"Overall reduction: {total_original_cards - len(refined_cards)} cards ({reduction_percent:.1f}%)\n\n")
            
            f.write("STAGE-BY-STAGE BREAKDOWN:\n")
            f.write("-"*60 + "\n\n")
            
            # Stage 1 summary
            f.write("STAGE 1: Initial Refinement\n")
            stage1_data = refinement_data['stage_breakdown']['stage1_refinement']
            f.write(f"  Removed: {stage1_data['removed']} cards\n")
            f.write(f"  Merged: {stage1_data['merged']} cards\n")
            f.write(f"  Modified: {stage1_data['modified']} cards\n\n")
            
            # Stage 2 summary
            f.write("STAGE 2: Hint Addition\n")
            stage2_data = refinement_data['stage_breakdown']['stage2_hints']
            f.write(f"  Hints added: {stage2_data['hints_added']}\n\n")
            
            # Stage 3 summary
            f.write("STAGE 3: Grouping Optimization\n")
            stage3_data = refinement_data['stage_breakdown']['stage3_grouping']
            f.write(f"  Cards regrouped: {stage3_data['regrouped']}\n\n")
            
            # Stage 4 summary
            f.write("STAGE 4: Ambiguity Check\n")
            stage4_data = refinement_data['stage_breakdown']['stage4_ambiguity']
            f.write(f"  Cards modified for clarity: {stage4_data['modified']}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("DETAILED DECISIONS:\n")
            f.write("-"*60 + "\n\n")
            
            # Write detailed decisions by stage
            for stage_num, stage_name in enumerate(['refinement', 'hints', 'grouping', 'ambiguity_check'], 1):
                stage_decisions = [d for d in all_decisions if d.get('stage') == stage_name]
                if stage_decisions:
                    f.write(f"\nSTAGE {stage_num} - {stage_name.upper()}:\n")
                    for decision in stage_decisions:
                        f.write(f"  Index: {decision.get('card_index', 'N/A')}\n")
                        f.write(f"  Action: {decision.get('action', 'N/A')}\n")
                        f.write(f"  Reason: {decision.get('reason', 'N/A')}\n")
                        f.write("  " + "-"*30 + "\n")
        
        print(f"📄 Human-readable summary saved to: {summary_file}")
        
        self.logger.info(f"Multi-stage refinement complete:")
        self.logger.info(f"  Stage 1: {stage1_data['removed']} removed, {stage1_data['merged']} merged, {stage1_data['modified']} modified")
        self.logger.info(f"  Stage 2: {stage2_data['hints_added']} hints added")
        self.logger.info(f"  Stage 3: {stage3_data['regrouped']} cards regrouped")
        self.logger.info(f"  Stage 4: {stage4_data['modified']} cards clarified")
    
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
                x = self.process_lecture(str(pdf_file), output_dir, resume=resume, budget_mode=budget_mode)
                successful += 1
                
                if x: 
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
    flex_mode = "--flex-processing" in sys.argv #OpenAI flex mode pricing
    
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
        add_hints=add_hints,
        flex_mode=flex_mode,
    )
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        generator.process_lecture(path, resume=resume, budget_mode=budget_mode)
    elif os.path.isdir(path):
        generator.process_folder(path, resume=resume, budget_mode=budget_mode)
    else:
        print("❌ Please provide a valid PDF file or folder path")


if __name__ == "__main__":
    main()