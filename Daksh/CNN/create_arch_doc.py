from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

doc = Document()

title = doc.add_heading('River Ice Classification — Architecture Discussion', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 1. 1D CNN
# ══════════════════════════════════════════════════════════════
doc.add_heading('1. 1D CNN (Current Model)', level=1)

doc.add_paragraph(
    'The 1D CNN treats each labeled pixel as a single sample with 12 numerical features. '
    'The input is reshaped from (batch, 12) to (batch, 1, 12) — treating the 12 features '
    'as a 1D signal with one channel — so Conv1d can slide a kernel across them and learn '
    'local feature interactions (e.g. how I1 and I2 relate for NDWI).'
)

doc.add_heading('Input Features (12)', level=2)
features = [
    ('I1, I2, I3, I4, I5', 'VIIRS I-band reflectances / brightness temperatures'),
    ('SZA, SAA, VZA, VAA', 'Solar and view zenith/azimuth angles'),
    ('VIIRS_NDWI',          'Derived: (I1 − I2) / (I1 + I2)'),
    ('water_fraction',      'MODIS-derived water fraction'),
    ('modis_ndvi',          'MODIS NDVI'),
]
ft = doc.add_table(rows=1, cols=2)
ft.style = 'Table Grid'
ft.rows[0].cells[0].text = 'Feature'
ft.rows[0].cells[1].text = 'Description'
for feat, desc in features:
    row = ft.add_row()
    row.cells[0].text = feat
    row.cells[1].text = desc

doc.add_paragraph()
doc.add_heading('Architecture', level=2)
arch0 = [
    'Input: (batch, 1, 12)',
    '        ↓',
    'Conv1d(1→32, kernel=3, padding=1)  +  BatchNorm  +  ReLU',
    'Conv1d(32→64, kernel=3, padding=1)  +  BatchNorm  +  ReLU',
    'AdaptiveMaxPool1d  +  Dropout(0.3)',
    '        ↓',
    'Flatten  →  256',
    'Linear(256→128)  +  ReLU  +  Dropout(0.3)',
    'Linear(128→4)   →  4 class logits',
]
p = doc.add_paragraph()
p.add_run('\n'.join(arch0)).font.name = 'Courier New'

doc.add_heading('Training Setup', level=2)
training = [
    ('Dataset',    '364 samples  |  291 train / 73 test (80/20 stratified)'),
    ('Optimizer',  'Adam  (lr=1e-3, weight_decay=1e-4)'),
    ('Scheduler',  'CosineAnnealingLR  (T_max=150)'),
    ('Loss',       'CrossEntropyLoss with inverse-frequency class weights'),
    ('Epochs',     '150 with early stopping on val accuracy (10% of train held out)'),
    ('CV',         '5-fold stratified cross-validation'),
]
tt = doc.add_table(rows=1, cols=2)
tt.style = 'Table Grid'
tt.rows[0].cells[0].text = 'Setting'
tt.rows[0].cells[1].text = 'Value'
for k, v in training:
    row = tt.add_row()
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
rt = doc.add_table(rows=1, cols=2)
rt.style = 'Table Grid'
rt.rows[0].cells[0].text = 'Metric'
rt.rows[0].cells[1].text = 'Value'
for k, v in results:
    row = rt.add_row()
    row.cells[0].text = k
    row.cells[1].text = v

doc.add_paragraph()
doc.add_heading('Per-Class Performance', level=2)
per_class = [
    ('ice_free / snow_free_land',       '0.81', '0.73', 'Strongest signal — open water + no snow'),
    ('ice_covered / snow_covered_land', '0.80', '0.74', 'Consistent high-reflectance snow signal'),
    ('ice_free / snow_land',            '0.73', '1.00', '100% recall — class weights worked on minority class'),
    ('ice_covered / snow_free_land',    '0.69', '0.75', 'Hardest — spectrally ambiguous, confused both ways'),
]
pct = doc.add_table(rows=1, cols=4)
pct.style = 'Table Grid'
pct.rows[0].cells[0].text = 'Class'
pct.rows[0].cells[1].text = 'F1'
pct.rows[0].cells[2].text = 'Recall'
pct.rows[0].cells[3].text = 'Notes'
for cls, f1, rec, note in per_class:
    row = pct.add_row()
    row.cells[0].text = cls
    row.cells[1].text = f1
    row.cells[2].text = rec
    row.cells[3].text = note

doc.add_paragraph()
doc.add_heading('Why It Struggles', level=2)
for lim in [
    '364 samples is small — RF outperforms on raw accuracy (78% vs 77%).',
    'No spatial context — each pixel classified independently, cannot see land background.',
    'ice_covered/snow_free sits spectrally between two other classes, confused from both sides.',
    'water_fraction and modis_ndvi may not be available at full-scene inference time.',
]:
    doc.add_paragraph(lim, style='List Bullet')

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 2. 2D CNN
# ══════════════════════════════════════════════════════════════
doc.add_heading('2. 2D CNN (Patch-Based)', level=1)

doc.add_paragraph(
    'Instead of a single pixel with 12 values, extract a 7×7 spatial patch around '
    'each labeled pixel. The model learns from the neighbourhood context — '
    'directly addressing the hardest problem of distinguishing land background '
    '(snow vs no snow) which a single pixel cannot capture.'
)

doc.add_heading('Input', level=2)
doc.add_paragraph('(batch, 12 bands, 7×7 pixels)')

doc.add_heading('Architecture', level=2)
arch1 = [
    'Input patch: (batch, 12 bands, 7×7 pixels)',
    '        ↓',
    'Conv2d(12→32, kernel=3, padding=1)',
    'BatchNorm → ReLU',
    'Conv2d(32→32, kernel=3, padding=1)',
    'BatchNorm → ReLU',
    'MaxPool2d(2×2)   →  (batch, 32, 3, 3)',
    '        ↓',
    'Conv2d(32→64, kernel=3, padding=1)',
    'BatchNorm → ReLU',
    'Conv2d(64→64, kernel=3, padding=1)',
    'BatchNorm → ReLU',
    'AdaptiveAvgPool2d(1×1)  →  (batch, 64, 1, 1)',
    '        ↓',
    'Flatten  →  64',
    'Linear(64→128)  →  ReLU',
    'Dropout(0.4)',
    'Linear(128→4)   →  4 class logits',
]
p = doc.add_paragraph()
p.add_run('\n'.join(arch1)).font.name = 'Courier New'

doc.add_heading('Why 7×7 patch', level=2)
for point in [
    'Small enough to keep training viable with 364 center pixels.',
    'Large enough to capture land background context.',
    'Each center pixel gets ~3 pixels of neighbourhood on each side.',
]:
    doc.add_paragraph(point, style='List Bullet')

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 3. U-Net
# ══════════════════════════════════════════════════════════════
doc.add_heading('3. U-Net (Full Scene Segmentation)', level=1)

doc.add_paragraph(
    'U-Net classifies every pixel in the entire scene in one forward pass — '
    'no sliding window needed. The encoder compresses spatial information while '
    'the decoder reconstructs the full resolution classification map. '
    'Skip connections preserve fine spatial detail lost during downsampling.'
)

doc.add_heading('Architecture', level=2)
arch2 = [
    'Input: (batch, 12 bands, H, W)  ←  full TIF scene',
    '',
    'ENCODER (contracting path)',
    '  Conv 3×3 → BN → ReLU  ×2',
    '  MaxPool ↓  +  save skip connection',
    '  12 → 64 → 128 → 256 → 512',
    '',
    'BOTTLENECK',
    '  512 → 1024',
    '',
    'DECODER (expanding path)',
    '  Upsample 2× + concat skip connection',
    '  Conv 3×3 → BN → ReLU  ×2',
    '  1024 → 512 → 256 → 128 → 64',
    '',
    'OUTPUT',
    '  Conv 1×1 → 4 classes',
    '  Output: (batch, 4, H, W)  ←  full classified raster',
]
p = doc.add_paragraph()
p.add_run('\n'.join(arch2)).font.name = 'Courier New'

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 3. U-Net with Attention Gates (Recommended)
# ══════════════════════════════════════════════════════════════
doc.add_heading('3. U-Net with Attention Gates (Recommended)', level=1)

doc.add_paragraph(
    'Same as U-Net but with attention gates added to the skip connections. '
    'The model learns to focus on relevant regions (river pixels) and suppress '
    'irrelevant background (pure land). The water fraction mask can guide this '
    'attention — directly addressing the mixed pixel problem.'
)

doc.add_heading('Architecture', level=2)
arch3 = [
    'Input: (batch, 12 bands, H, W)',
    '',
    'ENCODER',
    '  12 → 64 → 128 → 256 → 512  (same as U-Net)',
    '',
    'BOTTLENECK',
    '  512 → 1024',
    '',
    'DECODER  +  ATTENTION GATES on skip connections',
    '  ┌─────────────────────────────────────────┐',
    '  │  Attention Gate                          │',
    '  │  g  (decoder signal)  ──┐               │',
    '  │  x  (encoder skip)   ──┤→ sigmoid gate  │',
    '  │                         └→ x * gate      │',
    '  └─────────────────────────────────────────┘',
    '  → model learns to weight river pixels higher',
    '  → suppresses pure land regions automatically',
    '',
    'OUTPUT',
    '  Conv 1×1 → 4 classes',
    '  Output: (batch, 4, H, W)',
]
p = doc.add_paragraph()
p.add_run('\n'.join(arch3)).font.name = 'Courier New'

doc.add_heading('Why attention gates help here', level=2)
for point in [
    'River pixels are a small fraction of any scene — attention stops the model being dominated by land pixels.',
    'Water fraction mask can initialise or guide the attention weights.',
    'Specifically helps distinguish land background (snow vs no snow) — the hardest classification problem.',
    'Practical with limited data — much less data-hungry than ViT.',
]:
    doc.add_paragraph(point, style='List Bullet')

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 4. Vision Transformer
# ══════════════════════════════════════════════════════════════
doc.add_heading('4. Vision Transformer (ViT) — Future Direction', level=1)

doc.add_paragraph(
    'ViT splits the input into fixed-size patch tokens and applies Transformer '
    'self-attention. Every token attends to every other token — capturing global '
    'context that CNNs miss with local kernels. State of the art for remote sensing '
    'classification but requires significantly more labeled data.'
)

doc.add_heading('Architecture', level=2)
arch4 = [
    'Input: (batch, 12 bands, H, W)',
    '        ↓',
    'Split into 16×16 patch tokens',
    '        ↓',
    'Linear projection → embedding dimension (e.g. 768)',
    '+ learnable positional encoding',
    '        ↓',
    'Transformer Encoder  ×12 layers',
    '  ┌──────────────────────────────┐',
    '  │  Layer Norm                  │',
    '  │  Multi-Head Self-Attention   │  ← every token attends to all others',
    '  │  Layer Norm                  │',
    '  │  Feed-Forward Network        │',
    '  └──────────────────────────────┘',
    '        ↓',
    '[CLS] token  →  Linear  →  4 class logits',
]
p = doc.add_paragraph()
p.add_run('\n'.join(arch4)).font.name = 'Courier New'

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 5. Comparison Table
# ══════════════════════════════════════════════════════════════
doc.add_heading('Comparison', level=1)

comp = [
    ('Architecture',     '1D CNN',               '2D CNN',           'U-Net + Attention',   'ViT'),
    ('Input',            'Single pixel (12)',     'Patch 7×7',        'Full scene',          'Full scene tokens'),
    ('Output',           'Per pixel class',       'Per patch class',  'Full raster',         'Per token class'),
    ('Data needed',      '~300  (done ✓)',         '~500+',            '~1,000+',             '10,000+'),
    ('Spatial context',  'None',                  'Local patch',      'Multi-scale + attn',  'Global attention'),
    ('Mixed pixels',     'Partial (feature only)','Partial',          'Best',                'Best'),
    ('Training time',    'Minutes',               'Fast',             'Medium',              'Long (GPU)'),
    ('Full scene infer', 'Pixel by pixel',        'Sliding window',   'One forward pass',    'Sliding window'),
    ('Current status',   'Trained ✓',             'Next step',        'Recommended path',    'Long term'),
    ('Test accuracy',    '76.71%',                'Est. 82-85%',      'Est. 85-88%',         'Est. 88-92%'),
]

t = doc.add_table(rows=len(comp), cols=5)
t.style = 'Table Grid'
for i, row_data in enumerate(comp):
    for j, val in enumerate(row_data):
        cell = t.rows[i].cells[j]
        cell.text = val
        if i == 0:
            cell.paragraphs[0].runs[0].bold = True

doc.add_paragraph()

# ── Pros / Cons per model ─────────────────────────────────────
doc.add_heading('Pros & Cons', level=1)

models_pc = [
    ('1D CNN  (current)',
     ['Works with 364 samples', 'Fast training — minutes on CPU', 'Simple deployment — one .pt file', 'Good balanced accuracy (78.46%) — handles minority classes', 'Already trained and working'],
     ['No spatial context — each pixel classified alone', 'Cannot see land background around the river', 'Needs modis_ndvi at inference time', 'RF outperforms on raw accuracy', 'Will not scale well with more data']),
    ('2D CNN  (next step)',
     ['Captures local spatial context (7×7 neighbourhood)', 'Directly sees land background — snow vs no snow', 'Still feasible with ~500 samples', 'Expected accuracy jump to 82-85%', 'Faster than U-Net for single-pixel queries'],
     ['Needs patch extraction from TIFs', 'Sliding window for full-scene inference (slow)', 'Patch size choice affects performance', 'Still limited to local context, not global']),
    ('U-Net + Attention Gates  (recommended)',
     ['Classifies full scene in one forward pass — no sliding window', 'Attention gates focus model on river pixels', 'Water fraction mask can guide attention directly', 'Multi-scale features — local and regional context', 'Well proven in remote sensing literature'],
     ['Needs ~1,000+ labeled pixels to train well', 'Longer training time', 'More complex to implement and debug', 'Needs GPU for reasonable training speed']),
    ('ViT  (long term)',
     ['Global attention — every pixel sees every other pixel', 'Best accuracy ceiling', 'Learns which bands matter per class via attention', 'State of the art for satellite imagery classification'],
     ['Needs 10,000+ labeled pixels — not viable now', 'Longest training time, requires GPU', 'Hardest to interpret', 'Overkill for current data size']),
]

for model_name, pros, cons in models_pc:
    doc.add_heading(model_name, level=2)
    doc.add_paragraph('Pros:', style='List Bullet').runs[0].bold = True
    for pro in pros:
        p = doc.add_paragraph(pro, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    doc.add_paragraph('Cons:', style='List Bullet').runs[0].bold = True
    for con in cons:
        p = doc.add_paragraph(con, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    doc.add_paragraph()

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 6. Roadmap
# ══════════════════════════════════════════════════════════════
doc.add_heading('Recommended Roadmap', level=1)

roadmap = [
    ('Now',       'Random Forest',           'Production inference on full TIF — fast, no GPU, 78% accuracy'),
    ('Next',      '2D CNN (7×7 patches)',     'Accuracy boost, captures local spatial context'),
    ('Then',      'U-Net + Attention Gates',  'Full scene segmentation, focuses on river pixels'),
    ('Long term', 'ViT / SegFormer',          'Best accuracy, requires 10,000+ labeled pixels'),
]
t2 = doc.add_table(rows=1, cols=3)
t2.style = 'Table Grid'
t2.rows[0].cells[0].text = 'Phase'
t2.rows[0].cells[1].text = 'Model'
t2.rows[0].cells[2].text = 'Reason'
for phase, model, reason in roadmap:
    row = t2.add_row()
    row.cells[0].text = phase
    row.cells[1].text = model
    row.cells[2].text = reason

doc.add_paragraph()
doc.add_paragraph(
    'The bottleneck is labeled data, not model architecture. '
    'With 364 samples, RF is best for production. '
    '2D CNN and U-Net become viable with more data. '
    'ViT is the long-term goal.'
)

out = Path('/Users/venesa/Desktop/CNN/outputs/Architecture_Discussion.docx')
doc.save(str(out))
print(f'Saved → {out}')
