# Code for “Can Audio Reveal Music Performance Difficulty? Insights From the Piano Syllabus Dataset”

This repository contains the official code and pretrained models for:

> **Can Audio Reveal Music Performance Difficulty? Insights From the Piano Syllabus Dataset**,  
> *IEEE Transactions on Audio, Speech and Language Processing*, 2025.  
> Pedro Ramoneda, Minhee Lee, Dasaem Jeong, Jose J. Valero-Mas, Xavier Serra.  
> DOI: [10.1109/TASLPRO.2025.3539018](https://doi.org/10.1109/TASLPRO.2025.3539018)

---

## 🔍 Quick Inference

To estimate difficulty from an audio file:

1**Run difficulty prediction using a pretrained model**:

```python predict_difficulty.py```

You can also try the composer_model , era_model, or multirank_model in place of the basic models.

---

## 📂 Repository Structure

```
├── models/
├── get_cqt.py
├── model.py
├── predict_difficulty.py
├── zero_shot.py
├── hidden_voices.json
├── hidden_voices_features.py
├── hidden_voices_logits.py
├── make_table_basic_model.py
├── make_table_composer_model.py
├── make_table_era_model.py
├── make_table_multirank_model.py
├── show_cqts.py
├── utils.py
├── poetry.lock
├── pyproject.toml
└── README.md
```
---

## 📊 Reproduce Paper Tables


```
python make_table_basic_model.py  
python make_table_composer_model.py  
python make_table_era_model.py  
python make_table_multirank_model.py
```

---

## 🧪 Zero-Shot Generalization (Hidden Voices)


```
python hidden_voices_features.py --config hidden_voices.json  
python hidden_voices_logits.py --config hidden_voices.json --models_dir models/  
python zero_shot.py`
```

---

## 📈 Visualize CQTs

`
python show_cqts.py --input cqts.npy --output cqt_plot.png
`
---

## 📋 How to Cite

```
@ARTICLE{10878288,
  author={Ramoneda, Pedro and Lee, Minhee and Jeong, Dasaem and Valero-Mas, Jose J. and Serra, Xavier},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={Can Audio Reveal Music Performance Difficulty? Insights From the Piano Syllabus Dataset},
  year={2025},
  volume={33},
  number={},
  pages={1129-1141},
  doi={10.1109/TASLPRO.2025.3539018}
}
```

---

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.