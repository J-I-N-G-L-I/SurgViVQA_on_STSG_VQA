# REAL-Colon-VQA Annotations

This folder contains the **annotation files** for the **REAL-Colon-VQA** dataset, a Video Visual Question Answering (VQA) benchmark built from real colonoscopic recordings.

REAL-Colon-VQA extends the original [REAL-Colon](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) with temporally aligned contextual and instrument-level metadata, enabling temporal and diagnostic reasoning in colonoscopy.
Each annotated clip consists of **8 frames** (stride = 4) extracted from 6 full colonoscopic procedures.

## 📁 Files

- **`in_template.jsonl`** – QA pairs following the *in-template* question style (aligned with the model’s training prompt template).  
- **`out_template.jsonl`** – QA pairs following the *out-template* question style (template-free, natural phrasing).

Each line in these `.jsonl` files represents one annotated QA sample.

## 📄 Annotation Format

Each annotation entry follows this structure:

```json
{
  "id": "qa_000007",
  "video_id": "002-003",
  "frames": [
    "002-003_39453", "002-003_39457", "002-003_39461",
    "002-003_39465", "002-003_39469", "002-003_39473",
    "002-003_39477", "002-003_39481"
  ],
  "frame_numbers": [39453, 39457, 39461, 39465, 39469, 39473, 39477, 39481],
  "question": "Is clean mucosa visible in this sequence?",
  "question_type": "mucosa_visibility",
  "support_frames": 8,
  "total_frames": 8,
  "support_ratio": 1.0,
  "support_percentage": 100.0,
  "answer": "No, clean mucosa is not consistently visible.",
  "short_answer": "no"
}
```

## Field Descriptions

| Field | Description |
|-------|--------------|
| `id` | Unique identifier of the QA pair. |
| `video_id` | Identifier of the source colonoscopy video. |
| `frames` | List of frame identifiers corresponding to an 8-frame sequence. |
| `frame_numbers` | Original frame indices within the colonoscopy video. |
| `question` | Natural language question related to the video sequence. |
| `question_type` | One of the 18 predefined question categories. |
| `support_frames`, `total_frames` | Number of frames providing visual support vs. total frames. |
| `support_ratio`, `support_percentage` | Normalized fraction of frames used as support. |
| `answer` | Long-form, descriptive answer. |
| `short_answer` | Short categorical answer (`yes`, `no`, etc.). |


## Domains & Question Types

REAL-Colon-VQA covers **6 reasoning domains**, each with multiple question categories (18 in total):

- **Instruments** – e.g., `instrument_presence`, `instrument_position`
- **Sizing** – e.g., `lesion_size`, `polyp_diameter`
- **Diagnosis** – e.g., `mucosa_visibility`, `lesion_type`
- **Positions** – e.g., `endoscope_orientation`, `field_of_view`
- **Operation Notes** – e.g., `procedure_stage`, `withdrawal_phase`
- **Movement** – e.g., `scope_motion`, `camera_translation`

## Citation

If you use this code, please cite:

```
@misc{drago2025surgvivqa,
  title        = {SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding},
  author       = {Mauro Orazio Drago and Luca Carlini and Pelinsu Celebi Balyemez and Dennis Pierantozzi and Chiara Lena and Cesare Hassan and Danail Stoyanov and Elena De Momi and Sophia Bano and Mobarak I. Hoque},
  year         = {2025},
  eprint       = {2511.03325},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2511.03325},
  note         = {Under revision}
}

```

