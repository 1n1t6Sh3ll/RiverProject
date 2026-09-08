from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

doc = Document()

# ── Title ─────────────────────────────────────────────────────
title = doc.add_heading('River Ice Classification — Model Architecture', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — 1D CNN
# ══════════════════════════════════════════════════════════════
doc.add_heading('1D Convolutional Neural Network (Current Model)', level=1)

doc.add_heading('Overview', level=2)
doc.add_paragraph(
    'The 1D CNN treats each labeled pixel as a single sample with 12 numerical features '
    '(spectral bands, viewing geometry, and derived indices). The input is reshaped from '
    '(batch, 12) to (batch, 1, 12) — treating the 12 features as a 1D signal with one channel — '
    'so Conv1d can slide a kernel across them and learn local feature interactions.'
)

doc.add_heading('Input Features (12)', level=2)
features = [
    ('I1, I2, I3, I4, I5', 'VIIRS I-band reflectances / brightness temperatures'),
    ('SZA, SAA, VZA, VAA', 'Solar and view zenith/azimuth angles'),
    ('VIIRS_NDWI',          'Derived: (I1 − I2) / (I1 + I2)'),
    ('water_fraction',      'MODIS-derived water fraction'),
    ('modis_ndvi',          'MODIS NDVI'),
]
table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
table.rows[0].cells[0].text = 'Feature'
table.rows[0].cells[1].text = 'Description'
for feat, desc in features:
    row = table.add_row()
    row.cells[0].text = feat
    row.cells[1].text = desc

doc.add_paragraph()

doc.add_heading('Architecture', level=2)
arch = doc.add_paragraph()
arch.add_run('Input: (batch, 1, 12)\n').bold = True
doc.add_paragraph('→  Conv1d(1→32, kernel=3, padding=1)  +  BatchNorm  +  ReLU')
doc.add_paragraph('→  Conv1d(32→64, kernel=3, padding=1)  +  BatchNorm  +  ReLU')
doc.add_paragraph('→  AdaptiveMaxPool1d  +  Dropout(0.3)')
doc.add_paragraph('→  Flatten  →  Linear(256→128)  +  ReLU  +  Dropout(0.3)')
doc.add_paragraph('→  Linear(128→4 classes)  →  Logits')

doc.add_heading('Training Setup', level=2)
training = [
    ('Optimizer',   'Adam (lr=1e-3, weight_decay=1e-4)'),
    ('Scheduler',   'CosineAnnealingLR (T_max=150)'),
    ('Loss',        'CrossEntropyLoss with inverse-frequency class weights'),
    ('Epochs',      '150 with early stopping on validation accuracy'),
    ('Split',       '80% train / 20% test (stratified), 10% of train for validation'),
    ('CV',          '5-fold stratified cross-validation'),
]
t2 = doc.add_table(rows=1, cols=2)
t2.style = 'Table Grid'
t2.rows[0].cells[0].text = 'Setting'
t2.rows[0].cells[1].text = 'Value'
for k, v in training:
    row = t2.add_row()
    row.cells[0].text = k
    row.cells[1].text = v

doc.add_paragraph()

doc.add_heading('Results', level=2)
results = [
    ('Test accuracy',      '76.71%'),
    ('Balanced accuracy',  '78.46%'),
    ('5-fold CV',          '75.27% ± 4.67%'),
    ('Macro F1',           '0.738'),
    ("Cohen's Kappa",      '0.661'),
    ('Macro IoU',          '0.590'),
]
t3 = doc.add_table(rows=1, cols=2)
t3.style = 'Table Grid'
t3.rows[0].cells[0].text = 'Metric'
t3.rows[0].cells[1].text = 'Value'
for k, v in results:
    row = t3.add_row()
    row.cells[0].text = k
    row.cells[1].text = v

doc.add_paragraph()

doc.add_heading('Per-Class Performance', level=2)
per_class = [
    ('ice_free / snow_free_land',       '0.81', 'Strongest signal — open water + no snow'),
    ('ice_covered / snow_covered_land', '0.80', 'Consistent high-reflectance snow signal'),
    ('ice_free / snow_land',            '0.73', '100% recall — class weights worked'),
    ('ice_covered / snow_free_land',    '0.69', 'Hardest — spectrally ambiguous'),
]
t4 = doc.add_table(rows=1, cols=3)
t4.style = 'Table Grid'
t4.rows[0].cells[0].text = 'Class'
t4.rows[0].cells[1].text = 'F1'
t4.rows[0].cells[2].text = 'Notes'
for cls, f1, note in per_class:
    row = t4.add_row()
    row.cells[0].text = cls
    row.cells[1].text = f1
    row.cells[2].text = note

doc.add_paragraph()

doc.add_heading('Limitations', level=2)
for lim in [
    '364 training samples is small for a neural network — RF outperforms on raw accuracy.',
    'Hardest class (ice_covered/snow_free) sits spectrally between two other classes.',
    'water_fraction and modis_ndvi not always available at inference time.',
    '1D CNN cannot leverage spatial context — each pixel classified independently.',
]:
    doc.add_paragraph(lim, style='List Bullet')

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Vision Transformer
# ══════════════════════════════════════════════════════════════
doc.add_heading('Vision Transformer (ViT) — Future Direction', level=1)

doc.add_heading('Overview', level=2)
doc.add_paragraph(
    'Vision Transformer (ViT) applies the Transformer self-attention mechanism to image patches. '
    'Instead of convolutional filters, it splits the input into fixed-size tokens and learns '
    'global relationships between all tokens simultaneously. Every pixel attends to every other '
    'pixel — capturing long-range spatial context that CNNs miss with local kernels.'
)

doc.add_heading('Architecture', level=2)
vit_steps = [
    ('1. Patch tokenisation', 'Split input (batch, 12, H, W) into 16×16 patches → flatten each patch into a vector token'),
    ('2. Linear projection',  'Project each token to embedding dimension (e.g. 768)'),
    ('3. Positional encoding','Add learnable position embeddings so the model knows spatial order'),
    ('4. Transformer encoder','Stack of N layers, each with:\n  - Multi-head self-attention (every token attends to all others)\n  - Feed-forward network\n  - Layer normalisation'),
    ('5. Classification head','[CLS] token or global average → Linear → 4 class logits'),
]
t5 = doc.add_table(rows=1, cols=2)
t5.style = 'Table Grid'
t5.rows[0].cells[0].text = 'Step'
t5.rows[0].cells[1].text = 'Description'
for step, desc in vit_steps:
    row = t5.add_row()
    row.cells[0].text = step
    row.cells[1].text = desc

doc.add_paragraph()

doc.add_heading('Why ViT for Remote Sensing', level=2)
for point in [
    'Global attention captures river-land boundary context across the full scene.',
    'Handles multi-scale features — local spectral signature and regional snow patterns.',
    'Self-attention can learn which bands matter most for each class.',
    'State-of-the-art on satellite image classification benchmarks.',
]:
    doc.add_paragraph(point, style='List Bullet')

doc.add_heading('Comparison: 1D CNN vs ViT', level=2)
comp = [
    ('Input',           'Single pixel (12 features)',  'Image patch (12 bands × H × W)'),
    ('Spatial context', 'None — pixel only',           'Full patch + global attention'),
    ('Data needed',     '~300 samples (viable)',       '10,000+ samples'),
    ('Training time',   'Minutes',                     'Hours (GPU required)'),
    ('Accuracy (est.)', '75-78%',                      '85-90%+ with enough data'),
    ('Production',      'Fast, lightweight',           'Heavier, needs GPU'),
    ('Current status',  'Trained and deployed',        'Future direction'),
]
t6 = doc.add_table(rows=1, cols=3)
t6.style = 'Table Grid'
t6.rows[0].cells[0].text = 'Aspect'
t6.rows[0].cells[1].text = '1D CNN'
t6.rows[0].cells[2].text = 'ViT'
for row_data in comp:
    row = t6.add_row()
    for i, val in enumerate(row_data):
        row.cells[i].text = val

doc.add_paragraph()

doc.add_heading('Recommended Roadmap', level=2)
roadmap = [
    ('Now',         'Random Forest — production inference on full TIF scenes'),
    ('Next',        '2D CNN with 7×7 patches — spatial context, modest data requirement'),
    ('Then',        'U-Net with attention gates — full scene segmentation in one pass'),
    ('Long term',   'ViT / SegFormer — best accuracy, requires 10,000+ labeled pixels'),
]
t7 = doc.add_table(rows=1, cols=2)
t7.style = 'Table Grid'
t7.rows[0].cells[0].text = 'Phase'
t7.rows[0].cells[1].text = 'Model'
for phase, model in roadmap:
    row = t7.add_row()
    row.cells[0].text = phase
    row.cells[1].text = model

doc.add_paragraph()
doc.add_paragraph(
    'The bottleneck is labeled data, not model architecture. '
    'With 364 samples, RF and 1D CNN are the appropriate tools. '
    'ViT becomes the right choice once data collection reaches thousands of labeled pixels.'
)

out = Path('/Users/venesa/Desktop/CNN/outputs/CNN_ViT_Architecture.docx')
doc.save(str(out))
print(f'Saved → {out}')
