
from train_model import NaiveBayesTextClassifier, load_dataset


def load_model():
    print("\nLoading training dataset...")

    texts, labels = load_dataset()

    print("Training AI model...")

    model = NaiveBayesTextClassifier()
    model.fit(texts, labels)

    print("Model trained successfully!")

    return model


def main():

    print("=" * 54)
    print("       AI-BASED FAKE NEWS DETECTION TOOL")
    print("       Built with Python - no scikit-learn")
    print("=" * 54)

    try:
        model = load_model()

    except Exception as error:
        print("\nError:", error)
        print("Please check whether news.csv and train_model.py exist.")
        return

    print("\nPaste a news article or headline below.")
    print("Type EXIT to close the application.\n")

    while True:

        article = input("Enter news text: ").strip()

        if article.lower() in ["exit", "quit"]:
            print("\nThank you for using Fake News Detection Tool!")
            break

        if len(article.split()) < 3:
            print("Please enter at least 3 words.\n")
            continue

        label, confidence = model.confidence(article)

        print("\n" + "-" * 45)
        print("       FAKE NEWS DETECTION RESULT")
        print("-" * 45)

        print("Prediction:", label)

        print(f"Model Confidence: {confidence * 100:.2f}%")

        if label == "FAKE":
            print("Warning: This news may be misleading.")

        else:
            print("This news matches the REAL training examples.")

        print("\nNote: This is a demonstration model.")
        print("Always verify news using trusted sources.")

        print("-" * 45 + "\n")


if __name__ == "__main__":
    main()
