# data_loader.py
# --------------
# This file contains the sample documents that our RAG app will search over.
#
# In a real application, you might load documents from a database, PDF files,
# or a web API. For this project, we use a hardcoded list to keep things simple
# and focus on learning the RAG concepts rather than data management.
#
# We've chosen short paragraphs on CS and AI topics so the content is
# relevant and interesting for students learning about these technologies.


# A list of short paragraphs covering various tech topics.
# These are the "documents" that will be stored in our vector database.
SAMPLE_DOCUMENTS = [
    # Python
    "Python is a high-level programming language known for its clean, readable syntax. "
    "It uses indentation to define code blocks and emphasizes simplicity, making it one of "
    "the most beginner-friendly languages. Python is widely used in web development, data "
    "science, automation, and artificial intelligence.",

    "Python's standard library includes modules for almost everything: working with files, "
    "making HTTP requests, parsing JSON, handling dates, and much more. This 'batteries included' "
    "philosophy means you can build powerful programs without installing many extra packages.",

    "Python was created by Guido van Rossum in the late 1980s while he was working at the Centrum "
    "Wiskunde & Informatica (CWI) in the Netherlands. He began the project in December 1989 and "
    "named the language after the British comedy group Monty Python's Flying Circus, not the snake. "
    "Guido wanted a language that was easy to read, powerful, and fun to use. The first public "
    "release, Python 0.9.0, came out in February 1991. Python 2.0 was released in 2000 and "
    "Python 3.0 in 2008, which cleaned up the language while keeping its core philosophy of "
    "readability and simplicity. Guido van Rossum served as Python's Benevolent Dictator For Life "
    "(BDFL) until stepping down from that role in 2018; the language is now guided by the Python "
    "Software Foundation and its community of developers.",

    "Anaconda is an open-source distribution of Python designed for data science, machine learning, "
    "and AI development. While Python is a programming language, Anaconda is a curated platform that "
    "bundles Python with tools for package management, environment management, and reproducible "
    "workflows. It was co-founded by Peter Wang and Travis Oliphant to make data science easier to "
    "set up and deploy at scale.",

    "Anaconda Distribution includes Python, Conda (a cross-platform package and environment manager), "
    "and Anaconda Navigator (a desktop GUI for managing environments and launching tools like Jupyter "
    "Notebooks and Spyder). It comes pre-installed with hundreds of popular data science libraries "
    "such as NumPy, pandas, Matplotlib, SciPy, and scikit-learn, so users can start analyzing data "
    "without manually installing dependencies. Miniconda is a smaller alternative that includes only "
    "Conda, Python, and essential packages — users can add more packages later with conda install.",

    # Machine Learning
    "Machine learning is a branch of artificial intelligence where systems learn from data "
    "instead of being explicitly programmed. A machine learning model finds patterns in training "
    "data and uses those patterns to make predictions or decisions on new, unseen data.",

    "Supervised learning is the most common type of machine learning. You train a model on "
    "labeled examples (input-output pairs), and it learns to predict the output for new inputs. "
    "Examples include image classification, spam detection, and house price prediction.",

    "A neural network is a machine learning model loosely inspired by the human brain. "
    "It consists of layers of interconnected nodes called neurons. Data flows through the layers, "
    "and the network learns by adjusting the strength of connections between neurons.",

    # Natural Language Processing
    "Natural Language Processing (NLP) is the field of AI focused on enabling computers to "
    "understand, interpret, and generate human language. Tasks include translation, sentiment "
    "analysis, summarization, and question answering.",

    "A transformer is a neural network architecture that revolutionized NLP. Models like GPT "
    "and BERT are built on transformers. They use a mechanism called 'attention' to weigh how "
    "important each word in a sentence is relative to every other word.",

    "Word embeddings are a way to represent words as vectors of numbers. Similar words end up "
    "close together in vector space. For example, 'king' and 'queen' would have similar vectors, "
    "as would 'dog' and 'puppy'. This lets computers understand word similarity mathematically.",

    # Databases
    "A relational database stores data in tables with rows and columns, similar to a spreadsheet. "
    "Tables are linked by relationships using foreign keys. SQL (Structured Query Language) is used "
    "to query and manipulate the data. Examples include PostgreSQL, MySQL, and SQLite.",

    "A vector database is a specialized database designed to store and search vector embeddings. "
    "Instead of exact keyword matches, it finds the most similar vectors using distance calculations. "
    "This makes it ideal for semantic search, recommendation systems, and RAG applications.",

    "NoSQL databases store data in formats other than tables — such as documents (MongoDB), "
    "key-value pairs (Redis), or graphs (Neo4j). They are often more flexible and scalable "
    "than relational databases for certain types of unstructured data.",

    # APIs & Cloud
    "An API (Application Programming Interface) is a way for two programs to communicate. "
    "A REST API uses HTTP requests (GET, POST, PUT, DELETE) to exchange data, usually in JSON "
    "format. When you use a weather app, it's calling a weather service's API behind the scenes.",

    "Cloud computing lets you use computing resources — servers, storage, databases — over the "
    "internet instead of on your own hardware. Providers like AWS, Google Cloud, and Azure offer "
    "these services on demand, so you pay only for what you use.",

    "Large Language Models (LLMs) like GPT-4 and Gemini are AI models trained on massive amounts "
    "of text data. They can generate text, answer questions, write code, and summarize documents. "
    "They're accessed through APIs and can be fine-tuned for specific tasks.",

    # RAG & AI Systems
    "Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses by first "
    "retrieving relevant documents from a knowledge base, then providing those documents as "
    "context to the LLM when generating an answer. This reduces hallucinations and grounds "
    "the model in real, verifiable information.",

    "Semantic search finds results based on meaning rather than exact keyword matches. If you "
    "search for 'fast car', semantic search can also return results about 'quick automobile' "
    "because the meanings are similar. This is powered by vector embeddings.",

    # Software Development
    "Git is a version control system that tracks changes to your code over time. It lets you "
    "save snapshots (commits) of your project, create branches for new features, and merge "
    "changes from multiple contributors. GitHub is a platform for hosting Git repositories.",

    "A virtual environment in Python is an isolated space where you can install packages "
    "without affecting other projects. This prevents conflicts between projects that need "
    "different versions of the same package. You create one with 'python -m venv venv'.",

    "Data structures are ways of organizing data so it can be accessed and modified efficiently. "
    "Common structures include lists (ordered collections), dictionaries (key-value pairs), "
    "sets (unique values), and queues (first-in, first-out ordering).",

    "Software testing is the practice of verifying that your code works correctly. Unit tests "
    "check individual functions in isolation. Integration tests check how components work "
    "together. Writing tests helps catch bugs early and makes code easier to change safely.",

    # Python — fundamentals (expanded)
    "Python variables are names that refer to values in memory. Python uses dynamic typing, "
    "so you do not declare a type explicitly — the interpreter infers it at runtime. Common "
    "built-in types include int (whole numbers), float (decimals), str (text), bool (True/False), "
    "list, tuple, dict, and set. You can check a value's type with the built-in type() function.",

    "Python functions are reusable blocks of code defined with the def keyword. Functions can "
    "accept parameters and return values using the return statement. Parameters can have default "
    "values, and functions can accept variable-length arguments with *args and **kwargs. "
    "Well-named functions with a single clear purpose make Python code easier to read and test.",

    "pip is Python's standard package installer. It downloads libraries from the Python Package "
    "Index (PyPI) and installs them into your active environment. Common commands include "
    "pip install package-name, pip list to see installed packages, and pip freeze to export "
    "dependencies for a requirements.txt file. Always use a virtual environment so project "
    "dependencies stay isolated from other projects on your machine.",

    "Object-oriented programming (OOP) in Python uses classes to bundle data (attributes) and "
    "behavior (methods) together. A class is a blueprint; an instance is a concrete object "
    "created from that class. Core OOP concepts include encapsulation (hiding internal details), "
    "inheritance (reusing behavior from a parent class), and polymorphism (different classes "
    "responding to the same method name in their own way).",

    # Machine Learning — expanded
    "Unsupervised learning finds patterns in data that has no labels. The algorithm explores "
    "structure on its own rather than learning input-output pairs. Common tasks include "
    "clustering (grouping similar records, such as customer segments) and dimensionality "
    "reduction (compressing many features into fewer while preserving important variation). "
    "K-means clustering and Principal Component Analysis (PCA) are widely used techniques.",

    "Reinforcement learning trains an agent to make sequential decisions in an environment. "
    "The agent takes actions, receives rewards or penalties, and learns a policy that maximizes "
    "long-term reward. Unlike supervised learning, there are no fixed correct labels for every "
    "step — the agent discovers good strategies through trial and error. Applications include "
    "game playing, robotics, and recommendation systems that adapt over time.",

    "Machine learning models are evaluated by splitting data into training, validation, and test "
    "sets. The training set teaches the model; the validation set tunes hyperparameters and "
    "detects overfitting; the test set gives an unbiased final score. Overfitting happens when "
    "a model memorizes training noise instead of learning general patterns, performing well on "
    "training data but poorly on new data. Regularization and more training data help reduce it.",

    "Classification predicts a category (spam vs not spam, cat vs dog), while regression predicts "
    "a continuous number (house price, temperature). Common classification metrics include accuracy, "
    "precision, recall, and F1 score. Regression models are often evaluated with mean absolute error "
    "(MAE) or root mean squared error (RMSE). Choosing the right metric depends on the business cost "
    "of false positives versus false negatives.",

    # Databases — expanded
    "SQL is the standard language for querying relational databases. SELECT retrieves columns from "
    "a table; WHERE filters rows by a condition; JOIN combines rows from related tables using keys. "
    "For example, SELECT name, email FROM users WHERE active = true returns active user records. "
    "Primary keys uniquely identify rows; foreign keys link tables together and enforce relationships.",

    "Database indexing speeds up lookups by maintaining a sorted structure (often a B-tree) for "
    "one or more columns. Without an index, the database may scan every row (a full table scan). "
    "Indexes help WHERE clauses and JOINs run faster but add storage overhead and slow down writes "
    "slightly because the index must be updated on INSERT and UPDATE. Index columns you filter or "
    "sort on frequently in production queries.",

    "ACID is a set of properties that make database transactions reliable: Atomicity (all steps "
    "succeed or all roll back), Consistency (rules and constraints stay valid), Isolation "
    "(concurrent transactions do not corrupt each other), and Durability (committed data survives "
    "crashes). Relational databases like PostgreSQL and MySQL provide ACID guarantees, which "
    "matters for financial records, inventory, and any system where partial updates are unacceptable.",

    # APIs — expanded
    "HTTP status codes tell clients whether an API request succeeded or failed. 200 means OK; "
    "201 means created; 400 means bad request (client error); 401 means unauthorized; 403 means "
    "forbidden; 404 means not found; 429 means rate limited; 500 means internal server error. "
    "APIs typically return JSON bodies with error messages alongside these codes so clients can "
    "handle failures gracefully.",

    "API authentication protects endpoints from unauthorized access. Common approaches include "
    "API keys (a secret string sent in a header), OAuth 2.0 (delegated access without sharing "
    "passwords), and JWT bearer tokens (signed tokens that encode identity and expiry). Never "
    "commit API keys to source control — store them in environment variables or a secrets manager "
    "and rotate keys if they are exposed.",

    "Rate limiting restricts how many API requests a client can make in a time window. It protects "
    "services from abuse, controls cost, and ensures fair usage across users. When limits are "
    "exceeded, servers often return HTTP 429 with a Retry-After header. Client code should "
    "implement exponential backoff — wait longer between each retry — rather than hammering the "
    "API immediately after a rate-limit error.",

    # AI concepts — expanded
    "Prompt engineering is the practice of writing clear instructions so an LLM produces useful "
    "output. Effective prompts specify the task, desired format, constraints, and examples when "
    "needed (few-shot prompting). Chain-of-thought prompting asks the model to explain its "
    "reasoning step by step, which often improves accuracy on complex questions. Prompting changes "
    "runtime behavior without retraining the model.",

    "Fine-tuning adapts a pre-trained model to a specific task by training on additional examples. "
    "It changes model weights and is best for consistent tone, classification, or domain-specific "
    "language patterns. RAG retrieves external documents at query time and does not retrain the "
    "model — it is better for factual, changing knowledge. Most production systems combine "
    "prompting and RAG; fine-tuning is added only when prompts alone cannot enforce the behavior.",

    "Embeddings map text into dense numerical vectors so meaning can be compared mathematically. "
    "Similar concepts have vectors that are close together in vector space. Cosine similarity and "
    "Euclidean (L2) distance are common ways to compare embeddings. In RAG, query and document "
    "embeddings power semantic search — finding relevant passages even when exact keywords differ.",

    "Generative AI models produce new content — text, code, or images — rather than only "
    "classifying input. Large language models (LLMs) predict the next token in a sequence, "
    "which enables fluent answers, summaries, and code generation. Tokens are the model's units "
    "of text (roughly word fragments); API pricing and context limits are often measured in tokens. "
    "Longer prompts and retrieved context consume more tokens and increase latency and cost.",
]


def get_documents():
    """
    Return the list of sample documents.

    Returns:
        A list of strings, each being a document to store in the vector database.
    """
    return SAMPLE_DOCUMENTS


def generate_ids(documents):
    """
    Generate a unique ID string for each document.

    ChromaDB requires each document to have a unique string ID.
    We simply use "doc_0", "doc_1", etc.

    Args:
        documents: The list of documents to generate IDs for.

    Returns:
        A list of ID strings like ["doc_0", "doc_1", ...].
    """
    return [f"doc_{i}" for i in range(len(documents))]
