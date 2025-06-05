# 🤖 Chatbot GenAI - Caso de Estudio PDI-UTP  

> **Este fork se basa en el repositorio original de [GenAIOps_Pycon2025](https://github.com/darkanita/GenAIOps_Pycon2025) de @darkanita.**

*(versión local con Ollama + RTX 4060 Ti)*

Este proyecto demuestra cómo construir, evaluar y automatizar un chatbot **RAG** (Retrieval Augmented Generation) siguiendo buenas prácticas de **GenAIOps**, **sin depender de la API de OpenAI**: todos los modelos (LLM + embeddings) se ejecutan en tu propia GPU mediante **Ollama**.

---

## 🧠 Caso de Estudio
El chatbot responde preguntas sobre el **Plan de Desarrollo Institucional** de la Universidad Tecnológica de Pereira a partir de documentos PDF internos.

---

## 📂 Estructura del proyecto
```
├── app/
│   ├── ui_streamlit.py           ← interfaz simple del chatbot
│   ├── main_interface.py         ← interfaz combinada con métricas
│   ├── run_eval.py               ← evaluación automática
│   ├── rag_pipeline.py           ← lógica de ingestión y RAG
│   └── prompts/
│       ├── v1_asistente_rrhh.txt
│       └── v2_resumido_directo.txt
├── data/pdfs/                    ← documentos fuente
├── tests/
│   ├── test_run_eval.py
│   ├── eval_questions_pdi.json         ← dataset de evaluación
├── .env.example
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
├── .github/workflows/
│   ├── eval.yml
│   └── test.yml
```

---

## 🚦 Ciclo de vida GenAIOps aplicado

### 1. 🧱 Preparación del entorno

```bash
# 1-a. clona el repo
git clone https://github.com/delany-ramirez/GenAI-Chatbot-PDI chatbot-pdi
cd chatbot-pdi

# 1-b. crea el entorno
conda create -n chatbot-pdi python=3.10 -y
conda activate chatbot-pdi
pip install -r requirements.txt      # incluye langchain-ollama

# 1-c. instala Ollama (solo una vez)
winget install --id Ollama.Ollama -e     # Windows
# curl -fsSL https://ollama.ai/install.sh | sh   # macOS / Linux

# 1-d. descarga los modelos locales
ollama pull llama3:8b           # LLM principal
ollama pull nomic-embed-text    # modelo de embeddings

# 1-e. variables opcionales
cp .env.example .env
# Abre .env y ajusta:
# OLLAMA_BASE_URL=http://localhost:11434
```

---

### 2. 🔍 Ingesta y vectorización

```bash
python -m app.rag_pipeline --rebuild_index
```

El script:

1. Carga PDFs de `data/pdfs/`.  
2. Trocea en *chunks* (512/50 por defecto).  
3. Genera *embeddings* con **`nomic-embed-text`** vía Ollama.  
4. Guarda un índice **FAISS** en `data/vectorstore/`.  
5. Registra parámetros en **MLflow**.

Parámetros personalizables:
```python
save_vectorstore(chunk_size=1024, chunk_overlap=100)
```

---

### 3. 🧠 Pipeline RAG

```python
from app.rag_pipeline import load_vectorstore_from_disk, build_chain
vectordb = load_vectorstore_from_disk()
chain    = build_chain(vectordb, prompt_version="v3_asistente_pdi")
```

* Usa **\`ChatOllama\`** como LLM y el FAISS como *retriever*.

---

### 4. 💬 Interfaz Streamlit

```bash
streamlit run app/ui_streamlit.py        # UI básica
# ó
streamlit run app/main_interface.py      # UI + métricas
```

---

### 5. 🧪 Evaluación automática

```bash
python app/run_eval.py
```

* Genera respuestas con el RAG local.  
* Evalúa con **LangChain Eval** (`QAEvalChain`) usando **ChatOllama**.  
* Registra métricas en **MLflow**.

---

### 6. 📈 Dashboard de resultados

```bash
streamlit run app/dashboard.py
```

---

### 7. 🔁 Automatización CI (GitHub Actions)

* `.github/workflows/eval.yml` – evaluación automática.  
* `.github/workflows/test.yml` – pruebas unitarias.

---

### 8. 🧪 Validación

```bash
pytest tests/test_run_eval.py
```
Exige ≥ 80 % de precisión con el dataset base.

---

## ⚙️ Stack tecnológico

| Capa              | Tecnología                                     |
|-------------------|------------------------------------------------|
| **LLM**           | \`llama3:8b\` (vía **Ollama**)                   |
| **Embeddings**    | \`nomic-embed-text\` (**Ollama**)                |
| **RAG framework** | **LangChain** + **FAISS**                      |
| **UI / MLOps**    | **Streamlit**, **MLflow**, **GitHub Actions**  |


---

## 🛠️ Requisitos de hardware

* GPU con ≥ 8 GB VRAM (RTX 4060 Ti 16 GB recomendada).  
* Driver NVIDIA 550+ y CUDA ≥ 12 (Ollama incluye cuBLAS).  
* El modelo `llama3:8b` consume ~7 GB; ajusta `OLLAMA_NUM_GPU_LAYERS` si necesitas limitar VRAM.

---

## 🎓 Desafío para estudiantes
🧩 Parte 1: Personalización

1. Elige un nuevo dominio
Ejemplos: salud, educación, legal, bancario, etc.

2. Reemplaza los documentos PDF
Ubícalos en data/pdfs/.

3. Modifica o crea tus prompts
Edita los archivos en app/prompts/.

4. Crea un conjunto de pruebas
En tests/eval_dataset.json, define preguntas y respuestas esperadas para evaluar a tu chatbot.

✅ Parte 2: Evaluación Automática

1. Ejecuta run_eval.py para probar tu sistema actual.
Actualmente, la evaluación está basada en QAEvalChain de LangChain, que devuelve una métrica binaria: correcto / incorrecto.

🔧 Parte 3: ¡Tu reto! (👨‍🔬 nivel investigador)

1. Mejora el sistema de evaluación:

    * Agrega evaluación con LabeledCriteriaEvalChain usando al menos los siguientes criterios:

        * "correctness" – ¿Es correcta la respuesta?
        * "relevance" – ¿Es relevante respecto a la pregunta?
        * "coherence" – ¿Está bien estructurada la respuesta?
        * "toxicity" – ¿Contiene lenguaje ofensivo o riesgoso?
        * "harmfulness" – ¿Podría causar daño la información?

    * Cada criterio debe registrar:

        * Una métrica en MLflow (score)

    * Y opcionalmente, un razonamiento como artefacto (reasoning)

    📚 Revisa la [documentación de LabeledCriteriaEvalChain](https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.criteria.eval_chain.LabeledCriteriaEvalChain.html) para implementarlo.

📊 Parte 4: Mejora el dashboard

1. Extiende dashboard.py o main_interface.py para visualizar:

    * Las métricas por criterio (correctness_score, toxicity_score, etc.).
    * Una opción para seleccionar y comparar diferentes criterios en gráficos.
    * (Opcional) Razonamientos del modelo como texto.    

🧪 Parte 5: Presenta y reflexiona
1. Compara configuraciones distintas (chunk size, prompt) y justifica tu selección.
    * ¿Cuál configuración genera mejores respuestas?
    * ¿En qué fallan los modelos? ¿Fueron tóxicos o incoherentes?
    * Usa evidencias desde MLflow y capturas del dashboard.

🚀 Bonus

- ¿Te animas a crear un nuevo criterio como "claridad" o "creatividad"? Puedes definirlo tú mismo y usarlo con LabeledCriteriaEvalChain.

---

¡Listo para ser usado en clase, investigación o producción educativa! 🚀