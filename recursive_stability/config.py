# config.py

MODEL_NAME = "mistral"  # change as needed
OLLAMA_URL = "http://localhost:11434/api/generate"

INITIAL_TEMPERATURE = 1.1
NUM_PROBLEMS = 200
NUM_RECURSIONS = 10

# Energy thresholds (can adjust)
TAU_SOFT = 0.45
TAU_MEDIUM = 0.63
TAU_HARD = 1.54

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DATABASE_PATH = "experiment.db"
