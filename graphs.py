import pandas as pd
import matplotlib.pyplot as plt
import os

def find_best_model_and_plot(prefix):
    fig, ax = plt.subplots()
    best_model = None
    best_acc = 0

    for repr_type in ['a', 'b', 'c', 'd']:
        file = f"model_{prefix}_{repr_type}.pt.accuracies.csv"
        if not os.path.exists(file):
            print(f"⚠ Missing file: {file}")
            continue

        df = pd.read_csv(file)
        # נניח שכל שורה מייצגת 512 משפטים — נצבור כמה משפטים ראינו ונחלק ב־100 לגרף
        df["x"] = df["num_sentences"].cumsum() / 100
        ax.plot(df["x"], df["accuracy"], label=f"repr {repr_type}")

        max_acc = df["accuracy"].max()
        if max_acc > best_acc:
            best_acc = max_acc
            best_model = repr_type

    ax.set_title(f"{prefix.upper()} Dev Accuracy")
    ax.set_xlabel("Sentences Seen (x100)")
    ax.set_ylabel("Dev Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_accuracy_plot.png")
    print(f"\n✅ Best {prefix.upper()} model: repr '{best_model}' with accuracy {best_acc:.4f}")
    return best_model

# הפעלת הפונקציות
best_pos = find_best_model_and_plot("pos")
best_ner = find_best_model_and_plot("ner")

# הדפסת פקודות חיזוי
print("\n📌 Now run the following commands to generate final predictions:")
print(f"python bilstmPredict.py {best_pos} model_pos_{best_pos}.pt pos/test > test4.pos")
print(f"python bilstmPredict.py {best_ner} model_ner_{best_ner}.pt ner/test > test4.ner")
