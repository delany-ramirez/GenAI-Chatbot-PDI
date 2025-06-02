import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_ollama import ChatOllama
from langchain.evaluation.qa import QAEvalChain

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3_asistente_pdi")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_questions_pdi.json"

# Cargar dataset
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = ChatOllama(
    model="llama3:8b",          # o el que prefieras
    temperature=0.0,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

langchain_eval = QAEvalChain.from_llm(llm)

# ✅ Establecer experimento una vez
mlflow.set_experiment(f"eval_{PROMPT_VERSION}_{DATASET_PATH}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

correct_count = 0

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Evaluación con LangChain
        graded = langchain_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        # 🔍 Imprimir el contenido real
        print(f"\n📦 Resultado evaluación LangChain para pregunta {i+1}/{len(dataset)}:")
        print(graded)

        lc_verdict = graded.get("value", "UNKNOWN")
        is_correct = graded.get("score", 0)

        if lc_verdict == "CORRECT":
            correct_count += 1   

        # Log en MLflow
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("dataset_path", DATASET_PATH)

        mlflow.log_metric("lc_is_correct", is_correct)

        print(f"✅ Pregunta: {pregunta}")
        print(f"🧠 LangChain Eval: {lc_verdict}")
        print(f"🤖 Respuesta correctas acumuladas: {correct_count} de {i+1}, Porcentaje: {correct_count/(i+1)}")


