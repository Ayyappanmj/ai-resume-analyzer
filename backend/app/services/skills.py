"""Skill taxonomy: category -> canonical skill -> match patterns (lowercase).

Patterns are matched on spaCy tokens (case-insensitive), so ambiguous words such as
"go" or "r" are deliberately only matched through unambiguous aliases.
"""

SKILL_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "Programming": {
        "Python": ["python", "python3"], "Java": ["java"], "JavaScript": ["javascript", "js", "es6"],
        "TypeScript": ["typescript"], "C++": ["c++", "cpp"], "C#": ["c#", "csharp"],
        "Go": ["golang"], "Rust": ["rust"], "Kotlin": ["kotlin"], "Swift": ["swift"],
        "PHP": ["php"], "Ruby": ["ruby"], "Scala": ["scala"], "SQL": ["sql"],
        "Bash": ["bash", "shell scripting"], "R": ["rstudio", "r programming", "r language"],
        "MATLAB": ["matlab"],
    },
    "Web": {
        "HTML": ["html", "html5"], "CSS": ["css", "css3"], "React": ["react", "react.js", "reactjs"],
        "Angular": ["angular"], "Vue": ["vue", "vue.js"], "Node.js": ["node.js", "nodejs"],
        "Django": ["django"], "Flask": ["flask"], "FastAPI": ["fastapi"], "Spring Boot": ["spring boot"],
        "Express": ["express.js", "expressjs"], "REST APIs": ["rest api", "rest apis", "restful", "restful apis"],
        "GraphQL": ["graphql"], "Tailwind": ["tailwind", "tailwindcss"], "Next.js": ["next.js", "nextjs"],
    },
    "Database": {
        "PostgreSQL": ["postgresql", "postgres"], "MySQL": ["mysql"], "MongoDB": ["mongodb", "mongo"],
        "Redis": ["redis"], "SQLite": ["sqlite"], "Elasticsearch": ["elasticsearch"],
        "Oracle": ["oracle"], "Cassandra": ["cassandra"], "DynamoDB": ["dynamodb"], "NoSQL": ["nosql"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"], "GCP": ["gcp", "google cloud"], "Azure": ["azure"],
        "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"], "Terraform": ["terraform"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous delivery"],
        "GitHub Actions": ["github actions"], "Jenkins": ["jenkins"], "Linux": ["linux"],
        "Git": ["git", "github", "gitlab"], "Ansible": ["ansible"], "Microservices": ["microservices", "microservice"],
    },
    "Data & ML": {
        "Machine Learning": ["machine learning", "ml"], "Deep Learning": ["deep learning"],
        "NLP": ["nlp", "natural language processing"], "Computer Vision": ["computer vision"],
        "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"], "scikit-learn": ["scikit-learn", "sklearn"],
        "Pandas": ["pandas"], "NumPy": ["numpy"], "Statistics": ["statistics", "statistical analysis"],
        "Data Analysis": ["data analysis", "data analytics"], "Data Visualization": ["data visualization", "data visualisation"],
        "Tableau": ["tableau"], "Power BI": ["power bi", "powerbi"], "Excel": ["ms excel", "microsoft excel", "excel spreadsheets", "advanced excel"],
        "Spark": ["spark", "pyspark"], "Airflow": ["airflow"], "ETL": ["etl"], "LLM": ["llm", "llms", "large language models"],
        "Hugging Face": ["hugging face", "huggingface", "transformers"], "Data Engineering": ["data engineering", "data pipelines"],
    },
    "Practices": {
        "Agile": ["agile"], "Scrum": ["scrum"], "Kanban": ["kanban"], "Testing": ["testing", "unit testing", "pytest", "junit"],
        "TDD": ["tdd", "test-driven development"], "System Design": ["system design", "software architecture", "design patterns"],
        "Code Review": ["code review", "code reviews"], "Debugging": ["debugging", "troubleshooting"],
        "Performance Optimization": ["performance optimization", "optimization", "scalability"],
        "Security": ["security", "cybersecurity", "oauth"], "Jira": ["jira"],
    },
    "Soft Skills": {
        "Communication": ["communication", "communicated"], "Leadership": ["leadership", "led", "leading"],
        "Teamwork": ["teamwork", "collaboration", "collaborated", "cross-functional"],
        "Problem Solving": ["problem solving", "problem-solving"], "Mentoring": ["mentoring", "mentored", "mentorship"],
        "Project Management": ["project management", "stakeholder management"], "Presentation": ["presentation", "presentations"],
    },
}

# pattern -> canonical, canonical -> category
PATTERN_TO_SKILL: dict[str, str] = {}
SKILL_TO_CATEGORY: dict[str, str] = {}
for _cat, _skills in SKILL_TAXONOMY.items():
    for _skill, _patterns in _skills.items():
        SKILL_TO_CATEGORY[_skill] = _cat
        for _p in _patterns:
            PATTERN_TO_SKILL[_p] = _skill
