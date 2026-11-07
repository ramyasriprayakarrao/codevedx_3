# AI-Based Fake News Detection Tool

A beginner-friendly NLP project that classifies news text as **FAKE** or **REAL** using a Multinomial Naive Bayes classifier implemented from scratch in Python.

## Important: no scikit-learn
This project does **not** import or use `sklearn`. It uses only Python standard-library modules (`csv`, `re`, `math`, `pickle`, and `collections`). No `pip install` is required.

## Project structure

```text
AI_Based_Fake_News_Detection/
├── main.py
├── train_model.py
├── preprocessing.py
├── news.csv
└── README.md
```

## Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open the extracted `AI_Based_Fake_News_Detection` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run these commands:

```powershell
py --version
py train_model.py
py main.py
```

If `py` is not recognized, use your installed Python executable directly, for example:

```powershell
& "C:\\Users\\ramay\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" train_model.py
& "C:\\Users\\ramay\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" main.py
```

## How it works

1. `news.csv` contains example news text and its label (`FAKE` or `REAL`).
2. `preprocessing.py` converts text to lowercase, removes URLs/punctuation, and splits it into words.
3. `train_model.py` learns word frequencies and class probabilities using Naive Bayes with Laplace smoothing.
4. The trained model is saved as `fake_news_model.pkl`.
5. `main.py` accepts text and displays a prediction with a model confidence score.

## Improve the model

The included CSV is a small demonstration dataset so the project runs immediately. It is **not** large or balanced enough for dependable real-world detection. Replace or expand it with a properly labelled, reliable dataset. Keep the CSV headers exactly `text,label`, and use only `FAKE` or `REAL` as labels. Avoid training and testing on the same examples when evaluating accuracy.

## Limitations

- The prediction is based on patterns in the training examples, not live fact-checking.
- Confidence is a model score, not the probability that a claim is actually true.
- A short or unfamiliar article may be classified incorrectly.
- Always verify important news with trusted independent sources.
