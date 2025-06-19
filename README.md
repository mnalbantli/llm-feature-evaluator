```
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

```

  llm-feature-evaluator/
    ├── main.py                         # Main script for execution
    ├── input\_features\_sample.csv       # Mock metadata input (no real data)
    ├── output\_structure\_example.csv    # Sample output for demonstration
    ├── .env.example                    # Example .env file structure
    ├── requirements.txt                # Python dependencies
    ├── .gitignore                      # Ignoring sensitive files
    ├── README.md                       # You're reading it!
    └── screenshots/
    └── terminal\_output.png         # Terminal log example

```

---

## 🧪 Example Output

![Terminal Screenshot](screenshots/terminal_output.png)

Example result (from `output_structure_example.csv`):

| Column Name     | Is Derived? | Used in Modeling? | Time-Dependent? | Notes |
|----------------|-------------|-------------------|------------------|-------|
| `Clicks`       | Yes         | Yes               | Yes              | High engagement metric, often bucketed. |

---

## 🔐 Environment Variables

You’ll need a `.env` file in your root directory:

```

DEEPSEEK\_API\_KEY=your\_actual\_api\_key\_here

````

> Never share this file publicly.  
> Instead, use `.env.example` to guide others on what to include.

---

## 📦 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````

### 2. Add your `.env` file

Create a `.env` file using the structure shown in `.env.example`.

### 3. Run the Tool

```bash
python main.py
```

---

## ❗ Data Privacy

This project **does not contain any real data**.
All examples are synthetic or anonymized for demonstration purposes.

---

## 👋 Author

Built by [Mustafa Nalbantli](https://linkedin.com/in/mustafanalbantli)
                           | Data Analyst | 

---

## ⭐️ Like this project?

* Give it a ⭐ on GitHub
* Fork and try it with your own dataset
* Message me on LinkedIn — happy to connect

```

