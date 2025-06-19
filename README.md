# 🧠 LLM Feature Evaluator

A Python-based tool that automates feature evaluation using an LLM (DeepSeek API).  
It helps data analysts and scientists reason about dataset columns and determine if they are suitable for modeling.

---

## 🚀 What It Does

For each column in your dataset, the tool answers:
- Is the feature derived?
- Should it be used in modeling?
- Does it require transformation?
- Is it time-dependent?
- Does a bucketed version exist?
- Does it have interaction potential?

It uses structured prompts built from your metadata and sends them to an LLM (DeepSeek) for explanation and classification.

---

## 📁 Project Structure

llm-feature-evaluator/
├── main.py # Main script for execution
├── input_features_sample.csv # Mock metadata input (no real data)
├── output_structure_example.csv # Sample output for demonstration
├── .env.example # Example .env file structure
├── requirements.txt # Python dependencies
├── .gitignore # Ignoring sensitive files
├── README.md # You're reading it!
└── screenshots/
└── terminal_output.png # Terminal log example

yaml
Kopyala
Düzenle

---

## 🧪 Example Output

![Terminal Screenshot](screenshots/terminal_output.png)

Example result (from `output_structure_example.csv`):

| Column Name     | Is Derived? | Used in Modeling? | Time-Dependent? | Notes                                     |
|----------------|-------------|-------------------|------------------|-------------------------------------------|
| `Clicks`       | Yes         | Yes               | Yes              | High engagement metric, often bucketed.   |

---

## 🔐 Environment Variables

Create a `.env` file in your root directory:

DEEPSEEK_API_KEY=your_actual_api_key_here

yaml
Kopyala
Düzenle

> ⚠️ Never share this file publicly.  
> ✅ Use `.env.example` to show the required structure.

---

## 📦 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Add your .env file
Create a .env file using the structure shown in .env.example.

3. Run the Tool
bash
Kopyala
Düzenle
python main.py
❗ Data Privacy
This project does not contain any real data.
All examples are synthetic or anonymized for demonstration purposes.

👋 Author
Built by Mustafa Nalbantlı
Data Analyst | LLM Automation Builder | Prompt Engineer

⭐️ Like this project?
Give it a ⭐ on GitHub

Fork and try it with your own dataset

Message me on LinkedIn — happy to connect

yaml
Kopyala
Düzenle

---

Let me know if you want a Turkish section at the end, or a header banner image suggestion for extra polish!








