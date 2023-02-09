from flask import Flask, request, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

def detect_hateful_content(text):
    # Initialize SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentiment_analyzer.polarity_scores(text)
    
    # Check the sentiment score
    if sentiment_score["compound"] < -0.5:
        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        if detect_hateful_content(text):
            return render_template("index.html", text=text, result="The text contains hateful content. Please revise before posting.")
        else:
            return render_template("index.html", text=text, result="The text is acceptable for posting.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
